console.log("TFJS Regression Projekt gestartet");

/* ============================================================
   DATEN
   ============================================================ */

function groundTruth(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateXValues(N = 100) {
    return Array.from({ length: N }, () => Math.random() * 4 - 2);
}

function computeYValues(xs) {
    return xs.map(groundTruth);
}

function gaussianNoise(mean = 0, variance = 0.05) {
    const u1 = Math.random();
    const u2 = Math.random();
    return mean + Math.sqrt(variance) *
        Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function addNoise(ys) {
    return ys.map(y => y + gaussianNoise());
}

function splitTrainTest(xs, ys) {
    const idx = xs.map((_, i) => i).sort(() => Math.random() - 0.5);
    return {
        train: {
            x: idx.slice(0, 50).map(i => xs[i]),
            y: idx.slice(0, 50).map(i => ys[i])
        },
        test: {
            x: idx.slice(50).map(i => xs[i]),
            y: idx.slice(50).map(i => ys[i])
        }
    };
}

const xs = generateXValues();
const ys_clean = computeYValues(xs);
const { train, test } = splitTrainTest(xs, ys_clean);

const ys_train_noisy = addNoise(train.y);
const ys_test_noisy = addNoise(test.y);

/* ============================================================
   PLOTS (WICHTIG: async!)
   ============================================================ */

async function plotDatasetClean() {
    await Plotly.newPlot('plot-clean', [
        { x: train.x, y: train.y, mode: 'markers', name: 'Train' },
        { x: test.x, y: test.y, mode: 'markers', name: 'Test' }
    ]);
}

async function plotDatasetNoisy() {
    await Plotly.newPlot('plot-noisy', [
        { x: train.x, y: ys_train_noisy, mode: 'markers', name: 'Train' },
        { x: test.x, y: ys_test_noisy, mode: 'markers', name: 'Test' }
    ]);
}

async function plotModel(id, dataY, model) {
    const xs = [];
    for (let x = -2; x <= 2; x += 0.01) xs.push(x);

    const ys = model.predict(tf.tensor2d(xs, [xs.length, 1])).dataSync();

    await Plotly.newPlot(id, [
        { x: train.x, y: dataY, mode: 'markers' },
        { x: xs, y: ys, mode: 'lines' }
    ]);
}

/* ============================================================
   MODEL
   ============================================================ */

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    return model;
}

function toTensors(xs, ys) {
    return {
        xs: tf.tensor2d(xs, [xs.length, 1]),
        ys: tf.tensor2d(ys, [ys.length, 1])
    };
}

/* ============================================================
   TRAINING
   ============================================================ */

async function trainCleanModel() {
    const model = createModel();
    const t = toTensors(train.x, train.y);

    await model.fit(t.xs, t.ys, { epochs: 200, batchSize: 32, verbose: 0 });
    return model;
}

async function trainBestFitModel() {
    const model = createModel();
    const t = toTensors(train.x, ys_train_noisy);

    await model.fit(t.xs, t.ys, { epochs: 150, batchSize: 32, verbose: 0 });
    return model;
}

async function trainOverfitModel() {
    const model = createModel();
    const t = toTensors(train.x, ys_train_noisy);

    await model.fit(t.xs, t.ys, { epochs: 1500, batchSize: 32, verbose: 0 });
    return model;
}

/* ============================================================
   LAZY LOADING
   ============================================================ */

let loaded = { a1: false, a2: false, a3: false, a4: false };

async function loadA1() {
    if (loaded.a1) return;

    await plotDatasetClean();
    await plotDatasetNoisy();

    loaded.a1 = true;
}

async function loadA2() {
    if (loaded.a2) return;

    const model = await trainCleanModel();
    await plotModel("plot-clean-train", train.y, model);

    loaded.a2 = true;
}

async function loadA3() {
    if (loaded.a3) return;

    const model = await trainBestFitModel();
    await plotModel("plot-best-train", ys_train_noisy, model);

    loaded.a3 = true;
}

async function loadA4() {
    if (loaded.a4) return;

    const model = await trainOverfitModel();
    await plotModel("plot-overfit-train", ys_train_noisy, model);

    loaded.a4 = true;
}

/* ============================================================
   UI
   ============================================================ */

function showSection(id) {
    document.querySelectorAll('.content-section')
        .forEach(sec => sec.style.display = sec.id === id ? 'block' : 'none');
}

/* ============================================================
   ACCORDION + LOADING
   ============================================================ */

document.querySelectorAll(".accordion-header").forEach((header, index) => {
    header.addEventListener("click", async () => {

        const item = header.parentElement;
        const content = header.nextElementSibling;

        item.classList.toggle("active");

        if (item.classList.contains("active")) {

            const loading = content.querySelector(".loading");
            if (loading) loading.style.display = "block";

            // wichtig: UI rendern lassen
            await new Promise(r => setTimeout(r, 50));

            if (index === 0) await loadA1();
            if (index === 1) await loadA2();
            if (index === 2) await loadA3();
            if (index === 3) await loadA4();

            // WARTET BIS PLOT WIRKLICH FERTIG IST
            await new Promise(r => requestAnimationFrame(r));

            if (loading) loading.style.display = "none";
        }
    });
});
