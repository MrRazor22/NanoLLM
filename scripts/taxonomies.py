BANKING_CLUSTERS = {
    "Card Delivery & Ordering": [
        "activate_my_card", "card_about_to_expire", "card_acceptance", "card_arrival",
        "card_delivery_estimate", "card_linking", "get_disposable_virtual_card",
        "get_physical_card", "getting_spare_card", "getting_virtual_card",
        "order_physical_card", "virtual_card_not_working", "visa_or_mastercard"
    ],
    "Card Security & Physical Issues": [
        "card_not_working", "card_swallowed", "change_pin", "compromised_card",
        "contactless_not_working", "lost_or_stolen_card", "lost_or_stolen_phone", "pin_blocked"
    ],
    "Card Payments & Charges": [
        "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
        "declined_card_payment", "extra_charge_on_statement", "pending_card_payment",
        "reverted_card_payment?", "transaction_charged_twice"
    ],
    "Cash & ATM Withdrawals": [
        "atm_support", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal", "pending_cash_withdrawal", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal"
    ],
    "Bank Transfers": [
        "beneficiary_not_allowed", "cancel_transfer", "declined_transfer", "failed_transfer",
        "pending_transfer", "receiving_money", "transfer_fee_charged", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing"
    ],
    "Account Settings & Identity": [
        "age_limit", "apple_pay_or_google_pay", "country_support", "edit_personal_details",
        "passcode_forgotten", "terminate_account", "unable_to_verify_identity",
        "verify_my_identity", "verify_source_of_funds", "why_verify_identity"
    ],
    "Top-up & Balance": [
        "automatic_top_up", "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "direct_debit_payment_not_recognised",
        "pending_top_up", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
        "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
        "topping_up_by_card", "verify_top_up"
    ],
    "Exchange & Currencies": [
        "exchange_charge", "exchange_rate", "exchange_via_app",
        "fiat_currency_support", "supported_cards_and_currencies"
    ],
    "Refunds & Spending Limits": [
        "Refund_not_showing_up", "disposable_card_limits", "request_refund"
    ]
}

MASSIVE_CLUSTERS = {
    "alarm": ["alarm_query", "alarm_remove", "alarm_set"],
    "audio": ["audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up"],
    "calendar": ["calendar_query", "calendar_remove", "calendar_set"],
    "cooking": ["cooking_query", "cooking_recipe"],
    "datetime": ["datetime_convert", "datetime_query"],
    "email": ["email_addcontact", "email_query", "email_querycontact", "email_sendemail"],
    "general": ["general_greet", "general_joke", "general_quirky"],
    "iot": ["iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on"],
    "lists": ["lists_createoradd", "lists_query", "lists_remove"],
    "music": ["music_dislikeness", "music_likeness", "music_query", "music_settings"],
    "news": ["news_query"],
    "play": ["play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio"],
    "qa": ["qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock"],
    "recommendation": ["recommendation_events", "recommendation_locations", "recommendation_movies"],
    "social": ["social_post", "social_query"],
    "takeaway": ["takeaway_order", "takeaway_query"],
    "transport": ["transport_query", "transport_taxi", "transport_ticket", "transport_traffic"],
    "weather": ["weather_query"]
}

