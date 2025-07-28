graph TD
    A["Audio Input"] --> B["Preprocessing / Feature Extraction"]
    B --> C["Anti-Spoofing Check"]
    C --> D["Speaker Verification Model"]
    D --> E["Result (Speaker, Confidence, Spoof Flag)"]
``` 