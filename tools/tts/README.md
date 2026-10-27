# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Note: it's highly recommended to always provide a speaker reference voice (via `--tts-speaker-file`); otherwise, the model's performance may be degraded.

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc.
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Qwen3-TTS

Available params:
- `--tts-lang` can be `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (default: `en`)
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3)

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

## Pocket TTS

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). It is required, the model produces almost no audio without it
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```

## Maya2

[maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native): a Qwen3 backbone predicts code 0 of each 12.5 Hz frame, a small depth decoder predicts codes 1..31, and the kyutai Mimi decoder turns the codes into 24 kHz audio.

Available params:
- `--tts-lang` can be `en`, `hi`, `bn`, `ta`, `te`, `kn`, `ml`, `mr`, `gu`, `pa`, `or`, `as` (default: `en`)
- pick a built-in voice by starting the prompt with `{Name}:`, for example `{Anirban}: Hello world` (default voice: `Ahana`)
- `--tts-speaker-file` is not used
- the backbone sampling (temperature 0.9, presence penalty, top-k 64) and the stop rule follow the reference engine, `--top-k` / `--temp` do not change them

Example usage:

```sh
llama-tts -m maya2-native.gguf     -mm mmproj-maya2-native.gguf     -p "{Vaidehi}: Hello world"     --tts-lang en     --output out.wav
```

**Note for GGUF conversion:**

The checkpoint does not contain the Mimi decoder, the `--mmproj` conversion fetches it from `kyutai/mimi` (a pinned revision) through `huggingface_hub`:

```sh
python convert_hf_to_gguf.py path/to/maya2-native --outfile maya2-native.gguf
python convert_hf_to_gguf.py path/to/maya2-native --mmproj --outfile mmproj-maya2-native.gguf
```
