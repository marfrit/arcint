# 1.1.0 — Bildgenerierung auf den Arc-Karten (Entwurf)

**Entwurf** (2026-10-17, Branch `imggen`). Nichts davon ist beschlossen. Der
Operator ändert und beschließt diese Charta; ein Agent tut das nicht. Jede
Aussage trägt ihre Evidenzklasse: `paper` (veröffentlichter Text, README,
Modellkarte), `code` (im Quelltext gelesen, Datei und Commit genannt),
`measured-here` (auf den Karten dieses Projekts gemessen), `Rechnung`
(abgeleitet, nicht gemessen) oder `decision` (Operator, datiert).

Ziel: Text-zu-Bild mit Diffusion-Transformern (DiT/MMDiT) auf dem B60 und dem
A770, auf derselben ggml-OpenCL-Grundlage und mit denselben Intel-Kerneln wie
die Sprachdienste. Die Referenz-Engine ist **stable-diffusion.cpp**
(<https://github.com/leejet/stable-diffusion.cpp>, Commit `f0aac52`, 2026-10-17, MIT).

## 0. Ausgangslage

### 0.1 Was der Bauplan voraussetzt, und was davon stimmt

Die Ausgangsthese (extern, 2026-10-17): Bildgenerierung ist rechnerisch ein
Prefill, kein Decode; DiT-Linear-Schichten und Attention sind dieselbe
Kernel-Familie wie arcints Prompt-Pfad; neu sind der VAE und das Gate.
Geprüft gegen den Quelltext:

| These | Befund | Klasse |
|---|---|---|
| Der DiT ist GEMM + Attention + RMSNorm/SiLU/GELU/RoPE | stimmt für Z-Image (`src/model/diffusion/z_image.hpp`), Qwen-Image (`qwen_image.hpp`), FLUX.2 (`flux.hpp`, mit FLUX.1 geteilt) | `code` |
| Kopfdimension 128 | Z-Image: 30 Köpfe x 128, hidden 3840 (`z_image.hpp:32-38`); Qwen-Image und FLUX: 128 | `code` (Z-Image), `paper` (die anderen) |
| Attention ohne kausale Maske, ohne KV-Cache | ja; sd.cpp ruft `ggml_flash_attn_ext` über `ggml_ext_attention_ext` (`src/core/ggml_extend.cpp:615, :713`), K/V nach F16, Akkumulation F32 | `code` |
| arcints DPAS-Attention passt | ja: `flash_attn_dpas` / `dpas2` nehmen Kopfdimension 128 oder 256, mehr als 8 Query-Zeilen, die Maske ist optional (`mask == nullptr` erlaubt), K/V f16 oder quantisiert (`ggml-opencl.cpp` mit der Serie, Auswahl vor `flash_attn_dpas2`) | `code` |
| `conv_2d`, `group_norm`, `upscale`, `concat`, `pad` fehlen in ggml-OpenCL | **falsch** für den Pin `3e4dfa8`: `supports_op` nimmt CONV_2D (f16/f32), IM2COL, GROUP_NORM (zusammenhängend), UPSCALE (nearest/bilinear, ohne Antialias), PAD (nicht zirkulär), CONCAT und sogar TIMESTEP_EMBEDDING. Ob sie auf Arc schnell sind, ist offen: es sind Upstreams auf Adreno zugeschnittene Kernel | `code` |
| Was wirklich fehlt | CONV_3D und IM2COL_3D (der Wan-VAE von Qwen-Image, `wan_vae.hpp`), CONV_2D_DW (nicht im Hauptpfad der drei Modelle). `custom_4d` steht nur im INT8-Pfad, den der Upstream-Modus abschaltet; `arange` rechnet sd.cpp schon auf der CPU vor (`flux.hpp:1038, :1539`) | `code` |
| "pp512 +43,6 %" der Prompt-Attention | steht nicht im Record dieses Projekts; nicht verwendet | — |
| stable-diffusion.cpp hat ein OpenCL-Backend | ja, aber nur als Durchreichung: `SD_OPENCL` setzt `GGML_OPENCL` (`CMakeLists.txt:123-126`), sd.cpp selbst enthält keinen OpenCL-Code; die Doku nennt OpenCL "Adreno only, Q4_0" (`docs/build.md:157-161`, `paper`); kein Bericht über Arc + OpenCL in den Issues | `code` / `paper` |

### 0.2 Was sd.cpp mitbringt (`code`, Commit `f0aac52`)

- **ggml** als Submodul auf einen Fork (`leejet/ggml`, `dc99877`, 13 Commits
  über ggml-org `a05c992` vom 2026-10-15). Die Fork-Commits berühren
  `ggml-opencl` nicht.
- **Upstream-Modus**: `-DSD_USE_UPSTREAM_GGML=ON -DSD_GGML_SOURCE_DIR=<Baum>`
  (`CMakeLists.txt:98-99`, `cmake/ggml.cmake:24-34`). Dann sind INT8 und
  SageAttention aus, FP8 wird beim Laden nach F16 gewandelt, RoPE nimmt den
  generischen Weg (`src/model/common/rope.hpp:1075`). Alle anderen
  `ggml_*`-Aufrufe gibt es upstream (Abgleich des Unteragenten gegen die
  ggml-org-Header, nicht gebaut).
- **`GGML_MAX_NAME=160`** (`CMakeLists.txt:326-329`); der Pin hat 64
  (`ggml/include/ggml.h:229`). Der Wert ändert das Layout von `ggml_tensor`:
  libllama und sd.cpp in einem Prozess brauchen dieselbe Übersetzung.
- **Modelle** (`src/model.h:14-71`): u. a. Z_IMAGE, QWEN_IMAGE (2.1), FLUX,
  FLUX2, FLUX2_KLEIN. Die Text-Encoder sind sd.cpps eigene LLM-Implementierung
  (`src/model/te/llm.hpp`): Qwen3 (Z-Image, FLUX.2 klein), Qwen2.5-VL
  (Qwen-Image), Mistral Small 3.2 (FLUX.2) (`conditioner.hpp:2002-2024`).
  Der Encoder lädt GGUF mit llama.cpp-Tensornamen (`name_conversion.cpp:136-154`).
- **VAE**: AutoEncoderKL (Z-Image, FLUX.2, 2D-Faltungen, GroupNorm mit 32
  Gruppen, nearest-Upscale); Wan-VAE (Qwen-Image, 3D-Faltungen).
  `--vae-tiling` (Kacheln 256x256, Überlappung 0,5), `--conv-direct auto`
  wählt per Probe zwischen `ggml_conv_2d_direct` und im2col + mul_mat.
- **Gewichte**: GGUF und safetensors; F16, BF16, Q4_0 bis Q8_0, K-Quants,
  IQ-Typen (`include/stable-diffusion.h:101-146`).
- **Dienst**: `sd-server` mit `/v1/images/generations` und `/v1/images/edits`
  (`examples/server/routes_openai.cpp:253-334`), A1111-API `/sdapi/v1/*` und
  eine asynchrone Job-API. Bibliothek: `new_sd_ctx`, `generate_image`
  (`stable-diffusion.h:518, :533`); der Server serialisiert einen Kontext
  mit einer Mutex.
- **Zufall**: das Rauschen entsteht auf der CPU (`--rng cuda` ist ein
  CPU-Port von Philox, Standard; `cpu` = MT19937), also bei fester Saat
  gleich über alle Backends. Daraus folgt das Gate in §3 (`code`, nicht
  gemessen).

### 0.3 Was arcints Kernel für den DiT schon tun (`code`, Serie 0001-0056)

- **GEMM**: Intel-XMX-Wege für Q4_K, Q5_K, Q6_K, die IQ-Typen
  (`ggml_cl_kq_type_index`) und F16/BF16-Gewichte (`kernel_mul_mm_f16w_f16`,
  Patch 0022). **Q8_0 hat keinen**: es läuft über Upstreams generischen
  Kernel. Q8_0 ist bei FLUX-GGUFs das übliche Format.
- **Attention**: s. o.; gebaut für Kopfdimension 256 (die Qwens) und 128
  (Mistral Small 24B); gemessen ist sie kausal, mit Maske, am Agenten (256).
- Die Shapes sind andere: M ≈ 4.096 Token je Schritt, keine Maske, andere
  Hidden-Größen. Die gesuchten Kachelformen (`kq_tune`, die GA-Läufe) sind
  für die Sprachmodelle gewählt.

### 0.4 Größenordnung (`Rechnung`, keine Messung)

- 1024 x 1024 Pixel, VAE-Faktor 8, Patches 2 x 2: 64 x 64 = **4.096
  Bild-Token** je Schritt, dazu die Text-Token.
- Z-Image (6B, `paper`): ~2 x 6e9 x 4.096 ≈ 49 TFLOP je Schritt ohne
  Attention. Der dichte 27B auf dem B60 prefillt bei 4k 787-837 t/s
  (`measured-here`), ≈ 43-45 TFLOP/s effektiv. Bei gleicher Ausnutzung ≈ 1,1
  s je Schritt, 8 Schritte (Turbo) ≈ 9 s, ohne Encoder und VAE.
- Speicher (Gewichte, grob): Z-Image BF16 ≈ 12 GB + Qwen3-4B Q8_0 ≈ 4,3 GB;
  Qwen-Image 20B Q4_K ≈ 11-12 GB + Qwen2.5-VL-7B Q4_K ≈ 4,7 GB; FLUX.2 dev
  32B Q4_K ≈ 18 GB + Mistral Small 3.2 Q4_K_M ≈ 14 GB (passt nicht zusammen
  auf eine Karte; s. D4).

## 1. Entscheidungen des Operators (offen)

| # | Frage | Optionen, mit dem Preis, soweit bekannt |
|---|---|---|
| D1 | Umfang. README und DESIGN §8.8 nennen Multimodalität als Nicht-Ziel (`decision`). Bilderzeugung ist Ausgabe, nicht Eingabe, ändert aber den Umfang von "LLM-Server" zu "Inferenzserver" | (a) 1.1.0 nimmt Text-zu-Bild auf; (b) ein Schwesterprojekt auf arcints ggml; (c) nur Kernel-Beiträge, der Dienst bleibt sd.cpp |
| D2 | Weg der Integration | (a) **sd-server** von sd.cpp, gebaut gegen arcints gepatchten ggml-Baum, als eigenes Paket und eigene Unit: am wenigsten Code, arcint liefert Kernel und Gates; (b) **arcint-Endpunkt** `/v1/images/generations` über libstable-diffusion im arcint-Prozess: eine Oberfläche, aber ein zweites Laufzeitsystem im Prozess, `GGML_MAX_NAME` muss für libllama mit; (c) eigener DiT-Pfad in arcint: verworfen, sd.cpp ist die Referenz und hat ihn schon |
| D3 | Karte und Unit | B60 auf Abruf (`Conflicts=` mit Agent, Writer, Flash-Next, wie der Writer) oder A770 (16 GiB, nur ohne den Coder); beides heißt: eine Karte je Prozess (DESIGN §3, README) |
| D4 | Erstes Modell | **Z-Image Turbo** (6B, 8 Schritte, Qwen3-Encoder, AutoEncoderKL ohne 3D-Faltung): kleinste Lücke. Danach Qwen-Image (braucht CONV_3D für den Wan-VAE) und FLUX.2 (32B + 24B-Encoder: Encoder und DiT nacheinander laden, oder Encoder auf der zweiten Karte, sd.cpp `--backend te=...`) |
| D5 | `GGML_MAX_NAME` | 160 für den ganzen Build (libllama mit; ein A/B nach §5 DESIGN muss zeigen, dass die Sprachdienste nicht leiden), oder 64 und sd.cpp auf 64 bauen (sd.cpp-Tensornamen sind länger: Kürzung prüfen) |
| D6 | Das Gate (§3): Toleranzen | Vorschlag in §3; der Operator setzt die Zahlen |

## 2. Kriterien für 1.1.0

### A. Es läuft

| # | Kriterium | Zustand |
|---|---|---|
| A1 | sd.cpp im Upstream-Modus baut gegen den Pin `3e4dfa8` mit der Serie (0001-0056 ohne 0031, 0055) und `GGML_OPENCL_USE_ADRENO_KERNELS=OFF` | offen |
| A2 | Das erste Modell (D4) erzeugt auf der gewählten Karte (D3) ein 1024x1024-Bild bei fester Saat, kein Op fällt still auf die CPU zurück: `GGML_SCHED_DEBUG` zählt die Splits, jede CPU-Insel ist benannt und entweder gebaut oder mit ihrem Zeitanteil begründet | offen |
| A3 | Die Sprachdienste bleiben unverändert: das Release-A/B (`tests/acceptance/ab.py`) gegen 1.0.0 besteht, die Prüfstand-Aufgabe ≥ 9/10 | offen |
| A4 | Ein Dienst: Unit, Paket, `/v1/images/generations` (OpenAI-Form), `/health`, `/props` mit Modell, Karte, Schritten, Sampler | offen |

### B. Es ist schnell

| # | Kriterium | Zustand |
|---|---|---|
| B1 | Profil je Op und Schritt auf der Karte (Gerätezeit, erster Schritt getrennt, da er Kernel baut) | offen |
| B2 | Jeder Kernel mit mehr als 10 % der Schrittzeit ist gegen seinen gemessenen Deckel gestellt (wie B4 der 1.0-Charta) oder als Kampagne offen | offen |
| B3 | Ein Vergleich mit einer Referenz auf Arc: sd.cpp mit Vulkan auf derselben Karte, gleiche Datei, gleiche Saat, gleiche Schritte | offen |

### C. Es ist richtig

| # | Kriterium | Zustand |
|---|---|---|
| C1 | Das Bild-Gate (§3) besteht für jedes ausgelieferte Modell | offen |
| C2 | Jeder neue oder geänderte Kernel hat seinen Red Case, der das Gate rot macht | offen |

## 3. Das Gate

Es gibt keine Logits, also kein KL. Die Entsprechung der Antwort-Ebene ist
das Bild; weil die Entrauschung kleine Fehler über die Schritte verstärkt,
reicht der Endvergleich zur Zuordnung nicht. Vorschlag (die Zahlen setzt der
Operator, D6):

- **Referenz**: sd.cpp auf der CPU in F32 (oder BF16), gleiche Datei, gleiche
  Saat, gleicher Sampler; das Rauschen ist backend-unabhängig (§0.2).
- **Je Schritt**: Kosinus-Ähnlichkeit und MSE der Latents gegen die Referenz,
  je Schritt protokolliert. Vergleichend wie beim KL-Bar: der Kandidat ist je
  Schritt höchstens um eine Toleranz schlechter als der Basis-Arm (das
  unveränderte ggml-OpenCL auf derselben Karte im selben Fenster).
- **Am Ende**: PSNR und SSIM (optional LPIPS) des decodierten Bildes gegen die
  Referenz, vergleichend gegen den Basis-Arm.
- **Inhalt**: ein fester Satz Prompts mit prüfbarem Inhalt (Gegenstand,
  Anzahl, Farbe, ein kurzer Schriftzug). Wie der Inhalt automatisch geprüft
  wird, ist offen: arcint lehnt Bildeingaben ab (DESIGN §8.8).
- **Red Case zuerst**: ein absichtlich verfälschter Kernel (z. B. eine
  vertauschte Kachel in der Attention) muss die Schritt-Kurve sichtbar
  wegdrücken.
- **Geschwindigkeit**: Sekunden je Schritt und je Bild bei 1024x1024, frischer
  Prozess je Zelle, erster Schritt getrennt; gegen das letzte Release im
  selben Fenster, wie bei den Sprachdiensten.

## 4. Phasen

1. **Phase 0, Bestand** (eine Kartenfenster-Sitzung, Ergebnis `measured-here`):
   sd.cpp im Upstream-Modus gegen arcints ggml-Baum bauen (A1); Z-Image Turbo
   auf der CPU als Referenz und auf der Karte; `GGML_SCHED_DEBUG`-Zählung der
   CPU-Inseln; Profil je Op (B1); dazu derselbe Lauf mit sd.cpp-Vulkan als
   Vergleich (B3). Ergebnis: die Liste der Kernel, nach Zeitanteil.
2. **Phase 1, Gate**: die Referenz-Captures, die Schritt-Kurven, der Red Case
   (§3). Vor jeder Kernelarbeit.
3. **Phase 2, Lücken schließen**, nach dem Profil, nicht nach Vermutung.
   Kandidaten aus dem Quelltext:
   - Q8_0-GEMM auf den XMX-Einheiten (heute generisch, §0.3), Vorbild die
     eigenen K-Quant-GEMMs;
   - die Attention ohne Maske bei M ≈ 4.096 (die Maskenlese entfällt; die
     Kachelform für nicht-kausal neu suchen, das Genom des GA bleibt,
     Fitness = Sekunden je Schritt);
   - CONV_2D und GROUP_NORM für den VAE, falls das Profil sie zeigt
     (`--vae-tiling` begrenzt den Speicher);
   - CONV_3D / IM2COL_3D erst mit Qwen-Image.
   Jede Änderung als nummerierter Patch der Serie mit README-Abschnitt,
   Messung und Red Case (DESIGN §1.1).
4. **Phase 3, Dienst** (nach D1-D3): Paket, Unit, Endpunkt, Gates in
   `tests/acceptance/cells.json`.
5. **Phase 4, weitere Modelle** (Qwen-Image, FLUX.2), je mit eigenem Gate.
6. **Release 1.1.0** nach dem bestehenden Ablauf (Review, Pre-Tag-Check,
   frischer Container, A/B gegen 1.0.0).

## 5. Risiken

- Upstreams OpenCL-Kernel für Faltung, GroupNorm und Upscale sind auf
  Adreno geschrieben; auf Arc ungemessen. Ein langsamer VAE kann die
  Schrittersparnis auffressen (`Rechnung`, Phase 0 klärt es).
- Die Text-Encoder laufen in sd.cpps eigener LLM-Implementierung, nicht in
  libllama: arcints MTP, Checkpoints und Lanes greifen dort nicht; die
  Kernel greifen, soweit es dieselben ggml-Ops sind.
- Quantisierung: Fehler integrieren sich über die Schritte; Q4_K für den DiT
  ist riskanter als beim LLM (`paper`, Erfahrungswerte aus der Community).
  Das Gate (§3) entscheidet je Datei.
- Speicher bei hoher Auflösung: die Attention wächst quadratisch in den
  Token (2048x2048 ≈ 16.384 Token, `Rechnung`); der VAE-Decoder braucht ohne
  Kacheln viel Aktivierungsspeicher.
- Zwei Laufzeitsysteme auf einem ggml: der Pin muss für beide passen; jeder
  Pin-Umzug prüft dann auch sd.cpp.

## 6. Referenzen

- stable-diffusion.cpp (`f0aac52`, <https://github.com/leejet/stable-diffusion.cpp>):
  `src/model/diffusion/{z_image,qwen_image,flux}.hpp`,
  `src/model/vae/{auto_encoder_kl,wan_vae}.hpp`, `src/model/te/llm.hpp`,
  `src/core/ggml_extend.cpp`, `examples/server/`.
- arcints Kernel: `contrib/llama.cpp/patches` (0008, 0017, 0037, 0038 für die
  Attention; 0001, 0009, 0013, 0022 für die GEMMs) und
  `contrib/llama.cpp/README.md`.
- Die Prompt-Attention des Agenten: `docs/campaigns/dense-prefill-attention.md`.
