import json
import os

from qiskit_ibm_runtime import QiskitRuntimeService


def get_creds(args):
    """Read IBM Quantum credentials from the config or a qiskit-ibm.json file.

    Args:
        args (dict): Config dict. Reads ``ibm_channel``, ``ibm_instance``,
            ``ibm_token``, ``ibm_url``, and optionally ``qiskit_json_path``
            pointing to a qiskit-ibm.json credentials file.

    Returns:
        dict: Credential dict with keys ``channel``, ``instance``, ``token``,
            ``url`` suitable for ``QiskitRuntimeService(**creds)``.
    """
    cred_source_dict = {
        "channel": "ibm_channel",
        "instance": "ibm_instance",
        "token": "ibm_token",
        "url": "ibm_url",
        "name": "name"
    }
    rval = {}
    account_name = args.get("name", None)
    for ibm_name, yaml_name in cred_source_dict.items():
        value = args.get(yaml_name, None)
        if value:
            rval[ibm_name] = value

    qiskit_json_path = args.get("qiskit_json_path", None)
    if qiskit_json_path:
        qiskit_json_path = os.path.expanduser(qiskit_json_path)
        if os.path.exists(qiskit_json_path):
            # load the qiskit json file
            with open(qiskit_json_path, "r") as jfile:
                creds = json.load(jfile)
            # Access keys and values
            # The items we want are actually in a nested dictionary, so we have to loop through the outer dictionary first, then the
            # nested one.  This nested dictionary (outer_value) is actually the value for the key in the parent dictionary.
            selected_creds = None
            if account_name:
                selected_creds = creds.get(account_name, None)
            elif len(creds) == 1:
                selected_creds = next(iter(creds.values()))

            if selected_creds:
                for ibm_name in cred_source_dict.keys():
                    if ibm_name not in rval:
                        value = selected_creds.get(ibm_name, None)
                        if value:
                            rval[ibm_name] = value
        else:
            print("IBM credentials not found. Verify the path to your qiskit-ibm.json file.")
    return rval


def instantiate_runtime_service(args):
    """Instantiate QiskitRuntimeService using credentials from ``get_creds``.

    Args:
        args (dict): Config dict passed to ``get_creds``.

    Returns:
        QiskitRuntimeService: Authenticated service instance.
    """
    return QiskitRuntimeService(**get_creds(args))
