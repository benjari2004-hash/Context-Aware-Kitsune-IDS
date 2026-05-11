def format_counterfactual(cf_dict):
    """
    Formats a counterfactual dict into human-readable text for the CSV column.

    If found=True:
       "If {feature} were {cf_value} (instead of {orig}),
        action would be {new_action} instead of {current}"

    If found=False:
       "No counterfactual found within search range"
    """
    return cf_dict["explanation"]
