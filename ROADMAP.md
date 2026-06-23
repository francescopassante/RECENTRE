# RECENTRE — Roadmap di miglioramento modelli

> Obiettivo: aumentare l'FD-gain (e la calibrazione di σ) rispetto alla baseline
> "previous-frame", coprendo al contempo i deliverable della proposal
> (TCN ✓, Transformer ✓, **CNN-Transformer hybrid**, **physics-informed**, **latency study**).

## Diagnosi del problema (perché questa roadmap è ordinata così)

- Predizione a **1 passo** (~0.72 s di TR HCP) di un segnale **quasi random-walk**:
  la baseline "non ti muovi" è fortissima.
- Il segnale utile sta **negli scatti** (jerk). FD-gain è negativo sui micro-movimenti
  e arriva a ~20% solo quando c'è moto vero.
- Solo **6 canali**, finestra cortissima (10 frame stride-2): **non è un problema di
  capacità del modello**. Il ritorno marginale di backbone più grossi è basso.
- ⇒ I guadagni veri vengono da **feature in ingresso, likelihood, termine FD,
  finestra/stride**. L'architettura conta meno (ma serve per lo studio comparativo).

Leggenda effort: 🟢 < mezza giornata · 🟡 1–2 giorni · 🔴 > 2 giorni
Leggenda impatto atteso: ⭐ basso · ⭐⭐ medio · ⭐⭐⭐ alto

---

## FASE 0 — Fondamentali (fai subito, basso rischio)

- [ ] **AdamW al posto di Adam** — `train.py:44`. Ora `Adam(weight_decay=1e-4)` applica
      una L2 accoppiata al momento adattivo, quasi inefficace. AdamW disaccoppia il decay.
      🟢 ⭐⭐ — *Loshchilov & Hutter, ICLR 2019, arXiv 1711.05101*
	(APPLICATO)
- [ ] **Dropout GRU 0.5 → {0.1, 0.2, 0.3}** — `configs/gru_generalist.yaml:10`. 0.5 in un
      GRU 2-layer agisce anche tra i layer ricorrenti + testa ⇒ sotto-parametrizzato. 🟢 ⭐⭐
	(NON TOCCO LA GRU)
- [ ] **TCN: kernel_size 2 → 3 e togli un blocco** — `configs/tcn_generalist.yaml:9-10`.
      Il campo recettivo (~31 frame) è già >> input (10): i blocchi dilation 8 (e parte 4)
      sono sprecati. Meno parametri, meno latenza. 🟢 ⭐⭐
	(APPLICATO)
- [ ] **Scheduler allineato all'obiettivo** — `engine.py:123`. Selezioni su `val_fdg` ma
      fai `scheduler.step(val_loss)`. Valuta cosine+warmup o step su `-val_fdg`. 🟢 ⭐
	(PICCOLA INCONSISTENZA, NON L'HO IMPLEMENTATA)
- [ ] **EMA dei pesi** (exponential moving average) durante il training. 🟢 ⭐
	(NON PENSO FACCIA MALE, MA DUBITO ABBIA UN IMPATTO)

**Exit criterion FASE 0:** GRU e TCN ri-allenati con i nuovi default, FD-gain ≥ baseline attuale.

---

## FASE 1 — Leve ad alto impatto (probabilmente > di qualsiasi nuovo modello)

- [ ] **Feature di velocità/accelerazione in input** — `dataset.py` (`TimeSeriesDataset`).
      Aggiungi differenze prime (e seconde) come canali extra: 6 → 12/18 canali
      (`input_dim` nei config va aggiornato). Il modello predice già un residuo (≈velocità):
      darglielo esplicito è l'inductive bias giusto per anticipare gli scatti. 🟡 ⭐⭐⭐
	(NON IMPLEMENTATO?
- [ ] **Likelihood Student-t al posto della Gaussiana** — `engine.py:46,69` + teste in
      `models.py` (aggiungi parametro ν, appreso o fisso ~3–5). Gli incrementi di moto sono
      a code pesanti: la Gaussiana paga gli spike gonfiando σ ovunque; la Student-t dà σ più
      affilata sulla parte liscia e calibrazione migliore sugli scatti. 🟡 ⭐⭐⭐
      *Salinas et al., DeepAR, Int. J. Forecasting 2020, arXiv 1704.04110*
- [ ] **Termine FD pesato sul moto** — `engine.py:75-77`. `fd_gain.mean()` include i
      micro-movimenti dove il guadagno è negativo per costruzione e trascina giù la media.
      Pesa per la magnitudine del moto (o applica sopra soglia). 🟢 ⭐⭐
	(SECONDO ME NON E' CORRETTO, IL MODELLO DEVE ESSERE MIGLIORE OVUNQUE)
- [ ] **Sweep finestra/stride** — `sequence_length` nei config + stride in `dataset.py`
      (ora fisso a 2 in `__getitem__` e `GPUBatchLoader`). Griglia
      `sequence_length ∈ {10,20,30} × stride ∈ {1,2}`. Stride-1 dà contesto recente denso
      (utile per beccare lo scatto a t+1). 🟡 ⭐⭐⭐
	(POSSIAMO AUMENTARE LA WINDOW MA NON LO STRIDE)
- [ ] **Noise-injection in training** (robustezza al rumore — asse del benchmark proposal).
      Già esiste `noise` in `metrics.evaluate()`; replicalo come augmentation in training. 🟢 ⭐⭐	(SI PUO PROVARE)

**Exit criterion FASE 1:** identificata la combinazione (feature × likelihood × finestra)
che massimizza FD-gain a parità di backbone.

---

## FASE 2 — Nuovi modelli (studio comparativo richiesto dalla proposal)

Ordine per rilevanza. ⚠️ = nato per long-horizon multivariate, sul nostro 1-passo
probabilmente NON batte GRU/TCN — includilo per completezza, non aspettarti il primo posto.

- [ ] **Conformer (CNN-Transformer hybrid)** — *richiesto dalla proposal.* Conv locale
      (scatti) + attention globale. Fusione naturale del tuo TCN+Transformer. 🔴 ⭐⭐⭐
      *Gulati et al., Interspeech 2020, arXiv 2005.08100*
- [ ] **Modello physics-informed (smoothness/jerk)** — *richiesto dalla proposal.* Non serve
      backbone nuovo: penalità sul **secondo differenziale** della traiettoria predetta
      (accelerazione/jerk limitati = inerzia della testa). Aggiungi termine in `engine.py`. 🟡 ⭐⭐⭐
      *Min-jerk: Flash & Hogan, J. Neurosci. 1985; PINN: Raissi et al., J. Comp. Phys. 2019*
- [ ] **State-space model (Mamba / S4)** — causale, **O(1) per-step in inference, streaming**:
      vince sull'asse latency/model-size (enfatizzato dalla proposal). 🔴 ⭐⭐
      *Gu & Dao, Mamba, arXiv 2312.00752; Gu et al., S4, ICLR 2022, arXiv 2111.00396*
- [ ] **iTransformer** ⚠️ — attention sui canali (i 6 DOF) invece che sul tempo: cattura
      l'accoppiamento fisico traslazioni↔rotazioni che i modelli channel-independent
      (DLinear/NLinear/PatchTST) ignorano. 🟡 ⭐
      *Liu et al., ICLR 2024, arXiv 2310.06625*
- [ ] **N-HiTS / N-BEATS** ⚠️ — decomposizione multi-rate, cugini "deep" di DLinear/NLinear. 🟡 ⭐
      *Challu et al., AAAI 2023, arXiv 2201.12886; Oreshkin et al., ICLR 2020, arXiv 1905.10437*
- [ ] **Deep ensemble** (3–5 GRU/TCN con seed diversi) per accuratezza + calibrazione. 🟢 ⭐⭐
      *Lakshminarayanan et al., NeurIPS 2017, arXiv 1612.01474*

Nuova architettura: classe in `models.py` + una riga nel dict `MODELS` + un config in `configs/`.

---

## FASE 3 — Benchmarking & deliverable (Task 3 della proposal)

- [ ] **Tabella latency / model-size** per ogni modello: **#param, MAC/inferenza,
      ms/step su CPU** (target real-time). Oggi manca nel repo. SSM/GRU vincono; l'attention
      O(T²) è penalizzata in teoria ma con T=10 è irrilevante — mostralo coi numeri. 🟡 ⭐⭐⭐
- [ ] **Robustezza al rumore**: curva FD-gain vs livello di rumore (usa `noise` in `evaluate`). 🟢 ⭐⭐
- [ ] **Stabilità temporale**: errore in funzione dell'orizzonte / drift su sequenze lunghe. 🟡 ⭐
- [ ] **Report comparativo finale**: accuracy · robustezza · stabilità · latency · size,
      con raccomandazione del modello migliore per il framework real-time RECENTRE.
- [ ] Aggiorna `README.md` + `assets/` **solo** se cambia la storia headline (vedi CLAUDE.md).

---

## Riferimenti modelli già nel repo

- DLinear / NLinear — Zeng et al., *Are Transformers Effective for TS Forecasting?*, AAAI 2023, arXiv 2205.13504
- PatchTST — Nie et al., ICLR 2023, arXiv 2211.14730
- TSMixer — Chen et al., TMLR 2023, arXiv 2303.06053
- TCN — Bai, Kolter, Koltun, arXiv 1803.01271

---

## Ordine consigliato (TL;DR)

1. FASE 0 (fondamentali) → ri-baseline GRU/TCN
2. FASE 1 feature velocità/accel + Student-t (ritorno più alto), poi finestra/stride
3. FASE 1 termine FD pesato sul moto
4. FASE 2: Conformer + penalità smoothness (richiesti), poi Mamba (storia latenza)
5. FASE 3: tabella latency/size + report comparativo
