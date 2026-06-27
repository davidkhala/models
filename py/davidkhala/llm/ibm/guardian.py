from typing import Optional

from ibm_watsonx_ai.foundation_models.moderations import Guardian
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.schema import GuardianDetectors


class Detectors:
    """
    `GuardianDetectors`:
    - `hap`: Hate, abuse, and profanity (HAP) filter
    - `pii`: Personal identifiable information (PII) filter
    """
    default = GuardianDetectors.get_sample_params()
    strict = {
        "hap": {"threshold": 0.1},
        "pii": {},
        "granite_guardian": {"threshold": 0.1},
    }


class Client:
    """
    https://www.ibm.com/docs/en/watsonx/saas?topic=prompts-filtering-model-content-ai-guardrails
    """

    detection_type = {
        'risk': 'hap',
        'pii': 'pii'
    }

    def __init__(self, api_client: APIClient, *, detectors: Optional[dict | GuardianDetectors] = None):
        if not detectors:
            detectors = Detectors.default
        self._ = Guardian(api_client=api_client, detectors=detectors)

    def detect(self, text: str):
        r = self._.detect(text)
        return [{
            "start": _["start"],
            "end": _["end"],
            'text': _['text'],
            'type': Client.detection_type[_['detection_type']] or _['detection_type'],
            'score': _['score'],
        } for _ in r['detections']]
