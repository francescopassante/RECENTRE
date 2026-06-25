# Prior fisici sul moto della testa — cosa abbiamo trovato

Indagine sui prior physics-informed richiesti dalla proposal. Tre interventi testati
empiricamente sui dict HCP (R+M+L, ~1.97M frame). Conclusione anticipata: i prior
"fisici classici" (smoothness, accoppiamento rigido) trovano poco margine; il segnale
sfruttabile è **temporale/oscillatorio**, non di smoothness.

## 1. Smoothness / accelerazione limitata — porta aperta, e dannoso sugli spike

Penalità sul 2° differenziale della traiettoria predetta (`engine.py: smoothness_penalty`,
attivabile con `lambda_smooth`). Critica:
- Il modello predice un **residuo** su frame liscio → non produce già accelerazioni assurde.
  Penalizzarle regolarizza verso ciò che fa già.
- Peggio: la FD-gain è dominata dagli **spike** (e dalla rotazione ×50 mm). Uno spike *è*
  un'accelerazione grande → la penalità combatte proprio i casi che contano. Effetto
  atteso sulla FD potenzialmente **negativo**.

Verdetto: tenere solo come **ablation negativa** ("provato il prior ovvio, non aiuta").

## 2. Accoppiamento rigido rotazione–traslazione (pivot del collo) — reale ma inutile dove serve

La testa è un corpo rigido che ruota attorno al collo → `t = P × θ` accoppia traslazioni e
rotazioni. Regressione T ~ rotazioni:

| | R² traslazione ~ rotazioni |
|---|---|
| **pose assolute** | Tx **0.38** (Rz, ~28 mm/rad), Ty 0.17, Tz 0.04 |
| **incrementi frame-to-frame** | Tx **0.08**, Ty 0.06, Tz 0.12 |

L'accoppiamento esiste sulla **posa assoluta** (coerente col pivot, offset antero-posteriore),
ma il modello predice **incrementi**, e lì spiega solo il 6–12%. Un altro prior marginale.

## 3. Autocorrelazione degli incrementi — il punto chiave

Quanto è prevedibile l'incremento `ΔX_t` dal passato? Autocorrelazione per lag (within-patient):

| dim | lag1 | lag2 | lag3 | lag4 |
|---|---|---|---|---|
| Tx | −0.02 | **−0.58** | −0.12 | +0.27 |
| Tz | +0.09 | **−0.43** | −0.20 | +0.13 |
| Ty | −0.04 | **−0.26** | −0.08 | +0.01 |
| Rx | +0.04 | −0.21 | −0.11 | +0.02 |
| Ry | −0.18 | −0.19 | −0.07 | +0.05 |
| Rz | −0.22 | −0.18 | −0.05 | +0.05 |

Lettura:
- **lag-1 ≈ 0** → a un passo l'incremento sembra rumore bianco (da cui la conclusione
  iniziale, sbagliata, "moto imprevedibile").
- **lag-2 fortemente negativo** (Tx −0.58) + lag-4 positivo → **oscillazione quasi-periodica
  (~periodo 4 frame ≈ 0.35 Hz, plausibilmente il respiro)**, anti-persistente a 2 passi. Gli
  incrementi **non** sono bianchi.

## 3b. Verifica sul campionamento stride-2 (quello che il modello usa) — il segnale è FORTE

Il check sopra è su incrementi grezzi (stride-1). Misurando direttamente sul campionamento
stride-2 del modello (input `…, t-4, t-2, t` → predire `t+1`):

| | Tx | Tz | Rz | Ry | Ty | Rx |
|---|---|---|---|---|---|---|
| autocorr step stride-2 (tra input consecutivi) | **−0.67** | −0.45 | −0.40 | −0.38 | −0.34 | −0.24 |
| corr( step da predire `X_{t+1}−X_t` , ultimo step `X_t−X_{t-2}` ) | **−0.43** | −0.23 | −0.31 | −0.29 | −0.22 | −0.12 |

- Gli step stride-2 fanno **zig-zag** (autocorr −0.67 su Tx): su → giù → su.
- E questo **predice il target**: l'ultimo step osservato anti-predice il prossimo
  (Tx −0.43, R² ≈ 0.19). **Non è rumore: è segnale sfruttabile**, ed è quasi certamente da
  qui che esce buona parte del 23% di FD-gain.
- Correzione rispetto alla bozza precedente: il margine sopra il baseline **non** è
  ~zero. Il lag-1 grezzo ≈ 0 ingannava; la struttura vive nell'oscillazione (lag-2 / step
  stride-2) ed è robusta.

## Implicazioni

1. È **questa struttura temporale** (anti-persistenza/oscillazione), non i prior di smoothness,
   il segnale che il modello sfrutta per il ~23% di FD-gain sul baseline previous-frame.
2. Lo **smoothness è il prior sbagliato**, non solo inutile: assume traiettoria liscia/persistente,
   ma i dati sono **anti-persistenti** (alta frequenza). Lo penalizzare l'accelerazione combatte
   il segnale vero. Per questo è stato rimosso dalla loss.
3. Il prior che invece "vede" questo è la **mean-reversion** (step recente su → prossimo giù) —
   ma il modello può già impararla dalla storia (le **feature di velocità** stride-2 = `X_t−X_{t-2}`
   sono esattamente il predittore anti-persistente, corr −0.43 sul target: è perché vel/acc aiutano).
4. **Direzione**: o **feature di velocità** (già operative su `main`, corrette a stride-2) o
   **architettura temporale** che cattura lo zig-zag/parità (Conformer, attention even/odd), o un
   **baseline mean-reversion** come riferimento più forte del previous-frame. Non altri prior lisci.

## Filo conduttore (3 esperimenti)
- **student-t** → pareggio (σ già calibrata)
- **smoothness** → porta aperta + combatte gli spike
- **accoppiamento rigido** → reale sulle pose, debole sugli incrementi

Il baseline è forte perché l'incremento a 1 passo è quasi-bianco; la struttura sfruttabile è
l'oscillazione a periodo ~4, temporale, non fisica-di-smoothness.
