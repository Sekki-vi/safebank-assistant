"""Banking tools for the SafeBank agents.

Every function takes a customer_id first. In protection that values comes from
the authenticated session, never from the model, so an agent can only touch
the signed-in customer's accounts."""

import os
import uuid
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

TABLE_NAME = os.environ.get("SAFEBANK_TABLE", "safebank-accounts")
REGION = os.environ.get("AWS_REGION", "us-east-1")
EASTERN = ZoneInfo("America/New_York")
CONFIRMATION_THRESHOLD = Decimal("500.00")
MAX_TRANSACTIONS = 50

_dynamodb = boto3.resource("dynamodb", region_name=REGION)
_table = _dynamodb.Table(TABLE_NAME)
_client = _dynamodb.meta.client

def _pk(customer_id):
    return f"CUSTOMER#{customer_id}"

def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def _mask(account_number):
    return f"****{account_number[-4:]}"

def _error(code, message):
    return {"ok": False, "error": code, "message": message}

def _list_accounts(customer_id):
    response = _table.query(
        KeyConditionExpression=Key("PK").eq(_pk(customer_id)) & Key("SK").begins_with("ACCOUNT#")   
    )
    return response["Items"]

def _find_own_account(accounts, reference):
    """Match one of the customer's accounts by full number, last four digits, or type."""
    reference = str(reference).strip().lower()
    matches = [
        account for account in accounts
        if account["account_number"] == reference
        or (len(reference) == 4 and account["account_number"].endswith(reference))
        or account["account_type"] == reference
    ]
    if not matches:
        return None, _error("account_not_found", "No matching account for this customer")
    if len(matches) > 1:
        return None, _error("account_ambiguous", "More than one account matches. Use the last four digits.")
    return matches[0], None

def _summary(account):
    return {
        "account": _mask(account["account_number"]),
        "type": account["account_type"],
        "balance": str(account["balance"]),
        "status": account["status"],
    }

def _parse_amount(amount):
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite() or value <= 0 or value.as_tuple().exponent < -2:
        return None
    return value.quantize(Decimal("0.01"))

def _external_sent_today(customer_id, account_number):
    """Total external transfers from this account since midnight Eastern."""
    start = datetime.combine(datetime.now(EASTERN).date(), time.min, EASTERN).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    response = _table.query(
        KeyConditionExpression=Key("PK").eq(_pk(customer_id)) & Key("SK").between(f"TXN#{account_number}#{_iso(start)}", f"TXN#{account_number}#{_iso(end)}")
    )
    return sum(
        (-item["amount"] for item in response["Items"] if item.get("category") == "transfer_external"), Decimal("0")
    )

def get_balance(customer_id: str, account: str | None = None) -> dict:
    """Get the currenmt balance of the customer's accounts.

    Args:
        customer_id: The signed-in customer. Supplied by the session, not the model.
        account: Optional last four digits, full account number, "checking", or "savings".
        Leave empty to return every account.
    """
    accounts = _list_accounts(customer_id)
    if not accounts:
        return _error("customer_not_found", "No accounts found for this customer.")
    if account:
        match, error = _find_own_account(accounts, account)
        if error:
            return error
        accounts = [match]
    return {
        "ok": True,
        "accounts": [_summary(a) for a in accounts]
    }

def get_transactions(customer_id: str, account: str, limit: int = 10) -> dict:
    """Get the most recent transactions for one of the customer's accounts, newest first.

    Args:
        customer_id: The signed-in customer. Supplied by the session, not the model.
        account: Last four digits, full account number, "checking", or "savings".
        limit: How many transactions to return, from 1 to 50.
    """
    match, error = _find_own_account(_list_accounts(customer_id), account)
    if error:
        return error
    try:
        limit = max(1, min(int(limit), MAX_TRANSACTIONS))
    except (TypeError, ValueError):
        return _error("invalid_limit", "limit must be a whole number from 1 to 50.")

    number = match["account_number"]
    response = _table.query(
        KeyConditionExpression=Key("PK").eq(_pk(customer_id)) & Key("SK").begins_with(f"TXN#{number}#"),
        ScanIndexForward=False,
        Limit=limit,
    )
    transactions = [
        {
            "date": item["date"],
            "description": item["description"],
            "category": item["category"],
            "amount": str(item["amount"]),
            "balance_after": str(item["balance_after"]),
        }
        for item in response["Items"]
    ]
    return {"ok": True, "account": _mask(number), "transactions": transactions}

def transfer_funds(
    customer_id: str,
    from_account: str,
    to_account: str,
    amount: str,
    idempotency_key: str,
    confirmed: bool = False,
) -> dict:
    """Transfer money from one of the customer's accounts.

    Transfers between the customer's own accounts post immediately with no limit.
    Transfers to anyone else are pending for 1 to 3 business days, count toward the
    daily limit, and need the customer's confirmation when over $500.

    Args:
        customer_id: The signed-in customer. Supplied by the session, not the model.
        from_account: Last four digits, full account number, "checking", or "savings".
        to_account: Another of the customer's accounts, or a full 10-digit account number.
        amount: Dollar amount as a string, for example "125.50".
        idempotency_key: Unique ID for this request. Retrying with the same key never sends twice.
        confirmed: True only after the customer explicitly confirmed the recipient and amount.
    """
    value = _parse_amount(amount)
    if value is None:
        return _error("invalid_amount", "Amount must be a positive dollar value with at most 2 decimal places.")
    if not idempotency_key or len(str(idempotency_key)) > 100:
        return _error("invalid_idempotency_key", "idempotency_key must be 1 to 100 characters.")

    pk = _pk(customer_id)
    idempotency_sk = f"IDEMPOTENCY#{idempotency_key}"
    previous = _table.get_item(Key={"PK": pk, "SK": idempotency_sk}).get("Item")
    if previous:
        return {**previous["result"], "duplicate": True}

    accounts = _list_accounts(customer_id)
    source, error = _find_own_account(accounts, from_account)
    if error:
        return error
    if source["status"] != "active":
        return _error("account_inactive", "Only active accounts can send transfers.")

    destination, _ = _find_own_account(accounts, to_account)
    to_reference = str(to_account).strip()
    if destination is None and not (len(to_reference) == 10 and to_reference.isdigit()):
        return _error("invalid_recipient", "Recipient must be one of your accounts or a full 10-digit account number.")

    from_number = source["account_number"]
    to_number = destination["account_number"] if destination else to_reference
    if from_number == to_number:
        return _error("same_account", "The source and destination accounts are the same.")

    internal = destination is not None
    if not internal:
        if value > CONFIRMATION_THRESHOLD and confirmed is not True:
            return {
                **_error(
                    "confirmation_required",
                    f"Ask the customer to confirm sending ${value} from {_mask(from_number)} to "
                    f"{_mask(to_number)}. Transfers cannot be cancelled once confirmed.",
                ),
                "requires_confirmation": True,
            }
        remaining = source["transfer_limit"] - _external_sent_today(customer_id, from_number)
        if value > remaining:
            return _error("daily_limit_exceeded", f"This exceeds the daily transfer limit. ${remaining} remains today.")
    if value > source["balance"]:
        return _error("insufficient_funds", "The account balance is too low for this transfer.")

    now = _iso(datetime.now(timezone.utc))
    transfer_id = f"T{uuid.uuid4().hex[:12].upper()}"
    new_balance = source["balance"] - value
    result = {
        "ok": True,
        "transfer_id": transfer_id,
        "status": "posted" if internal else "pending",
        "arrives": "immediately" if internal else "in 1 to 3 business days",
        "from": _mask(from_number),
        "to": _mask(to_number),
        "amount": str(value),
        "new_balance": str(new_balance),
    }

    def balance_update(account, balance):
        # balance = :old is optimistic locking, so a concurrent change cancels the whole transfer
        return {"Update": {
            "TableName": TABLE_NAME,
            "Key": {"PK": pk, "SK": f"ACCOUNT#{account['account_number']}"},
            "UpdateExpression": "SET balance = :new",
            "ConditionExpression": "balance = :old AND #status = :active",
            "ExpressionAttributeNames": {"#status": "status"},
            "ExpressionAttributeValues": {":new": balance, ":old": account["balance"], ":active": "active"},
        }}

    def transaction_record(account_number, description, category, signed_amount, balance_after):
        return {"Put": {
            "TableName": TABLE_NAME,
            "Item": {
                "PK": pk,
                "SK": f"TXN#{account_number}#{now}#{transfer_id}",
                "account_number": account_number,
                "date": now,
                "description": description,
                "category": category,
                "amount": signed_amount,
                "balance_after": balance_after,
                "transfer_id": transfer_id,
            },
            "ConditionExpression": "attribute_not_exists(PK)",
        }}

    items = [
        balance_update(source, new_balance),
        transaction_record(
            from_number, f"Transfer to {_mask(to_number)}",
            "transfer_internal" if internal else "transfer_external", -value, new_balance,
        ),
        {"Put": {
            "TableName": TABLE_NAME,
            "Item": {"PK": pk, "SK": idempotency_sk, "result": result, "created_at": now},
            "ConditionExpression": "attribute_not_exists(PK)",
        }},
    ]
    if internal:
        destination_balance = destination["balance"] + value
        items += [
            balance_update(destination, destination_balance),
            transaction_record(
                to_number, f"Transfer from {_mask(from_number)}",
                "transfer_internal", value, destination_balance,
            ),
        ]

    try:
        _client.transact_write_items(TransactItems=items)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "TransactionCanceledException":
            raise
        reasons = [reason.get("Code") for reason in exc.response.get("CancellationReasons", [])]
        if len(reasons) > 2 and reasons[2] == "ConditionalCheckFailed":
            previous = _table.get_item(Key={"PK": pk, "SK": idempotency_sk}).get("Item")
            if previous:
                return {**previous["result"], "duplicate": True}
        return _error("transfer_conflict", "The account changed during the transfer. No money was moved. Try again.")
    return result