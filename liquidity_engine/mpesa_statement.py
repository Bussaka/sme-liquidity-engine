"""Load and classify M-Pesa statements (personal PDF statements or M-PESA for Business CSV exports).

Output is one row per completed transaction:
    receipt, time, details, amount (+ paid in / - withdrawn), balance, category, family,
    counterparty (keyed hash, never the raw name or number), is_betting

`family` maps onto the owner model's seven transaction families; `category` keeps the
finer Kenyan detail (Fuliza, digital loans, savings, charges) used for extra signals
and labels. Classification rules are a plain table so they can be tuned against real
statements: check `unclassified_share()` on the first pilot files.
"""
import hashlib
import hmac
import os
import re

import numpy as np
import pandas as pd

FAMILIES = ['deposit', 'withdraw', 'received', 'mm_send', 'paybill', 'merchantpay', 'transfer_from_bank']

LENDERS = r'tala|branch|zenka|okash|timiza|hustler fund|kcb m-?pesa|m-?shwari loan|eazzy|opesa|zash|mkopo'
BANKS = (r'equity|kcb|co-?op|cooperative|ncba|stanbic|absa|family bank|dtb|diamond trust|i&m|standard chartered|'
         r'sidian|kingdom bank|credit bank|gulf african|national bank|prime bank|bank')
BETTING = r'sportpesa|betika|odibets|mozzart|betin|1xbet|shabiki|betway|sportybet|bet ?lion|22bet|helabet'

# (pattern on lower-cased details, category, family). First match wins.
RULES = [
    (r'reversal', 'reversal', None),
    (r'charge', 'charge', None),
    (r'overdraft of credit party|od loan disbursement|fuliza.*(disburs|draw)', 'fuliza_draw', None),
    (r'od loan repayment|fuliza.*repay|overdraw', 'fuliza_repay', None),
    (rf'(pay ?bill|merchant payment|buy goods).*({LENDERS})', 'loan_repayment', 'paybill'),
    (rf'(business payment|b2c|funds received).*({LENDERS})', 'digital_loan_in', 'received'),
    (r'm-?shwari (deposit|transfer to)|lock savings|to m-?shwari', 'savings_out', None),
    (r'm-?shwari withdraw|from m-?shwari', 'savings_in', None),
    (r'airtime|bundle', 'airtime', None),
    (r'deposit of funds at agent|cash deposit', 'deposit', 'deposit'),
    (r'withdrawal at agent|customer withdrawal|atm withdrawal|withdrawal charge-free', 'withdraw', 'withdraw'),
    (rf'(business payment|b2c|transfer|funds received).*({BANKS})|from bank|bank to m-?pesa', 'transfer_from_bank',
     'transfer_from_bank'),
    (r'pay ?bill|paybill', 'paybill', 'paybill'),
    (r'merchant payment|buy goods|customer payment to small business|pochi', 'merchantpay', 'merchantpay'),
    (r'customer transfer|send money|customer send', 'mm_send', 'mm_send'),
    (r'funds received|received from|business payment from|promotion payment|salary', 'received', 'received'),
]
_RULES = [(re.compile(p), c, f) for p, c, f in RULES]
_BETTING = re.compile(BETTING)
_COUNTERPARTY = re.compile(r'(?:\bto|\bfrom|\btill|\bagent)\s+(?P<id>[\w*]+)\s*-\s*(?P<name>[^,]*)', re.I)


INFLOW_CATEGORIES = {'deposit', 'received', 'transfer_from_bank', 'digital_loan_in', 'savings_in', 'fuliza_draw'}


def classify(details, amount):
    text = str(details).lower()
    for pattern, category, family in _RULES:
        if pattern.search(text):
            if category == 'transfer_from_bank' and amount < 0:
                return 'transfer_to_bank', None
            if category in ('reversal', 'charge', 'transfer_to_bank'):
                return category, family
            if (category in INFLOW_CATEGORIES) != (amount > 0):
                # Direction contradicts the description; surface it rather than guess
                return 'unclassified', None
            return category, family
    return 'unclassified', None


def counterparty_key(details, salt):
    """Keyed hash of the counterparty so raw names and numbers never leave the parser."""
    m = _COUNTERPARTY.search(str(details))
    ident = f'{m.group("id")}|{m.group("name").strip().lower()}' if m else ''
    if not ident:
        return None
    return hmac.new(salt, ident.encode(), hashlib.sha256).hexdigest()[:16]


# --- Loading -----------------------------------------------------------------------------

_ROW_START = re.compile(r'^(?P<receipt>[A-Z0-9]{10})\s+(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\s*(?P<rest>.*)$')
_STATUS = re.compile(r'\b(COMPLETED|FAILED|CANCELLED|PENDING|REVERSED)\b')
_AMOUNT = re.compile(r'-?\d[\d,]*\.\d{2}')


def parse_statement_text(text):
    """Parse the transaction table from extracted PDF text (rows may wrap over several lines)."""
    rows, current = [], None
    for line in text.splitlines():
        line = line.strip()
        m = _ROW_START.match(line)
        if m:
            if current:
                rows.append(current)
            current = {'receipt': m['receipt'], 'time': m['time'], 'text': m['rest']}
        elif current and line:
            current['text'] += ' ' + line
    if current:
        rows.append(current)

    out = []
    for r in rows:
        status = _STATUS.search(r['text'])
        if not status:
            continue
        details = r['text'][:status.start()].strip()
        nums = [float(x.replace(',', '')) for x in _AMOUNT.findall(r['text'][status.end():])]
        if len(nums) < 2:
            continue
        balance = nums[-1]
        if len(nums) >= 3:
            paid_in, withdrawn = nums[-3], nums[-2]
            amount = paid_in if paid_in else -abs(withdrawn)
        else:
            amount = nums[0]
        out.append({'receipt': r['receipt'], 'time': r['time'], 'details': details,
                    'status': status.group(1), 'amount': amount, 'balance': balance})
    return pd.DataFrame(out, columns=['receipt', 'time', 'details', 'status', 'amount', 'balance'])


def read_pdf(path, password=None):
    from pypdf import PdfReader
    reader = PdfReader(path)
    if reader.is_encrypted:
        if not password or not reader.decrypt(password):
            raise ValueError(f'{path}: encrypted statement, wrong or missing password')
    return parse_statement_text('\n'.join(page.extract_text() or '' for page in reader.pages))


def read_csv(path):
    """M-PESA for Business portal export (Receipt No., Completion Time, Details, Paid In, Withdrawn, Balance)."""
    df = pd.read_csv(path)
    df.columns = [re.sub(r'[^a-z]', '', c.lower()) for c in df.columns]
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0) if c in df else 0
    paid_in, withdrawn = num('paidin'), num('withdrawn')
    return pd.DataFrame({
        'receipt': df['receiptno'], 'time': df['completiontime'], 'details': df['details'],
        'status': df['transactionstatus'].str.upper() if 'transactionstatus' in df else 'COMPLETED',
        'amount': np.where(paid_in != 0, paid_in, -np.abs(withdrawn)), 'balance': num('balance'),
    })


def load_statement(path, password=None, salt=None):
    """Load one statement file into classified, pseudonymised transactions."""
    path = str(path)
    raw = read_pdf(path, password) if path.lower().endswith('.pdf') else read_csv(path)
    return prepare(raw, salt)


def prepare(raw, salt=None):
    salt = salt or os.environ.get('PILOT_SALT', '').encode()
    if not salt:
        raise ValueError('Set PILOT_SALT (a secret kept outside the repo) to pseudonymise counterparties')
    tx = raw[raw['status'].str.upper() == 'COMPLETED'].copy()
    tx['time'] = pd.to_datetime(tx['time'])
    tx[['category', 'family']] = pd.DataFrame([classify(d, a) for d, a in zip(tx['details'], tx['amount'])],
                                              index=tx.index)
    tx['is_betting'] = tx['details'].str.lower().str.contains(_BETTING)
    tx['counterparty'] = tx['details'].map(lambda d: counterparty_key(d, salt))
    tx = tx.drop(columns=['status', 'details'])
    return tx.drop_duplicates('receipt').sort_values('time').reset_index(drop=True)


def merge_statements(frames):
    """Combine several statement periods for one business, dropping overlapping receipts."""
    return pd.concat(frames).drop_duplicates('receipt').sort_values('time').reset_index(drop=True)


def unclassified_share(tx):
    """Share of transaction value the rules could not classify; tune RULES if this is high."""
    value = tx['amount'].abs()
    return float(value[tx['category'] == 'unclassified'].sum() / max(value.sum(), 1))
