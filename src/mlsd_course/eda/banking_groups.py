"""Ручная группировка интентов BANKING77 по тому, решается ли интент поиском в приложении.

ДОПУЩЕНИЕ: группировка сделана вручную по названиям и примерам интентов, а не взята из датасета.

- ``search`` — фича приложения, которую клиент может найти поиском (выпустить карту, сменить PIN).
- ``faq`` — справочный вопрос, на который поиск может показать раздел справки или тарифы.
- ``ticket`` — инцидент по конкретной операции («списали дважды», «перевод не дошёл»):
  поиском не решается, нужен чат или поддержка.
"""

from __future__ import annotations

SEARCH_INTENTS = [
    "activate_my_card",
    "apple_pay_or_google_pay",
    "automatic_top_up",
    "cancel_transfer",
    "card_about_to_expire",
    "card_linking",
    "change_pin",
    "edit_personal_details",
    "exchange_rate",
    "exchange_via_app",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "receiving_money",
    "terminate_account",
    "top_up_by_cash_or_cheque",
    "topping_up_by_card",
    "transfer_into_account",
    "verify_my_identity",
]

FAQ_INTENTS = [
    "age_limit",
    "atm_support",
    "card_acceptance",
    "card_delivery_estimate",
    "country_support",
    "disposable_card_limits",
    "exchange_charge",
    "fiat_currency_support",
    "supported_cards_and_currencies",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_limits",
    "transfer_timing",
    "visa_or_mastercard",
    "why_verify_identity",
]


def intent_group(intent: str) -> str:
    """search / faq / ticket. Всё, что не перечислено явно, считается тикетом."""
    if intent in SEARCH_INTENTS:
        return "search"
    if intent in FAQ_INTENTS:
        return "faq"
    return "ticket"
