# Poetry evaluation corpus

These five Qwen3-TTS VoiceDesign recordings are copied byte-for-byte from the Speaking MIDI demo's `examples/poetry` assets. All five are evaluated in full, without cropping, padding, or generating replacement speech.

| Recording | Author | Language | Duration |
|---|---|---|---:|
| [Quiet Night Thoughts / 静夜思](quiet-night.wav) | Li Bai | Chinese | 8.64 s |
| [Spring Dawn / 春晓](spring-dawn.wav) | Meng Haoran | Chinese | 8.16 s |
| [On the Stork Tower / 登鹳雀楼](stork-tower.wav) | Wang Zhihuan | Chinese | 7.60 s |
| [Sonnet 18, opening couplet](sonnet-18.wav) | William Shakespeare | English | 4.64 s |
| [The Tyger, first stanza](the-tyger.wav) | William Blake | English | 8.24 s |

The source texts are public domain. Exact texts and the requested voice description are in [prompts.json](prompts.json). Model revision, generation parameters, measured durations, and SHA-256 hashes are in [manifest.json](manifest.json).

See [the backtest report](../../experiments/POETRY_BACKTEST.md) for metrics, algorithm ablations, limitations, and reproduction instructions.
