# Real-World Test Log

**Example log output from a real user test:**

```
[INFO] Received audio file: user_test.wav
[INFO] Prediction time: 0.512 seconds
[INFO] End-to-end response time: 0.789 seconds
Result: {'speaker': 'user1', 'is_spoof': False, 'prediction_time': 0.512}
```

**TODO:** Add screenshots of test flow and logs from real user and spoofed audio tests. 
| OAF_back_angry.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 3.81s) | - |
| OAF_back_happy.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 3.52s) | - |
| OAF_back_neutral.wav | YAF_fear (Confidence: 28.00%, Time: 3.21s) | Reliability: LOW | 2025-07-15 14:59:07 |
| OAF_back_happy.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 0.08s) | Reliability: LOW | 2025-07-15 14:59:07 |
| OAF_back_angry.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 0.07s) | Reliability: LOW | 2025-07-15 14:59:08 |
| OAF_back_neutral.wav | YAF_fear (Confidence: 28.00%, Time: 3.33s) | Reliability: LOW | 2025-07-15 15:01:39 |
| OAF_back_happy.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 0.08s) | Reliability: LOW | 2025-07-15 15:01:39 |
| OAF_back_angry.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 0.11s) | Reliability: LOW | 2025-07-15 15:01:40 |
| OAF_back_sad.wav | YAF_pleasant_surprised (Confidence: 26.00%, Time: 0.09s) | Reliability: LOW | 2025-07-15 15:01:40 |
| OAF_back_neutral.wav | YAF_fear (Confidence: 28.00%, Time: 2.97s) | Reliability: LOW | 2025-07-15 15:18:19 |
| OAF_back_happy.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 0.09s) | Reliability: LOW | 2025-07-15 15:18:20 |
| OAF_back_neutral.wav | YAF_fear (Confidence: 28.00%, Time: 0.09s) | Reliability: LOW | 2025-07-15 15:18:28 |
| OAF_back_happy.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 0.12s) | Reliability: LOW | 2025-07-15 15:18:28 |
| OAF_back_angry.wav | YAF_pleasant_surprised (Confidence: 28.00%, Time: 0.09s) | Reliability: LOW | 2025-07-15 15:18:29 |
| OAF_back_sad.wav | YAF_pleasant_surprised (Confidence: 26.00%, Time: 0.09s) | Reliability: LOW | 2025-07-15 15:18:29 |