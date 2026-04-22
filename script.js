console.log("TFJS Regression Projekt gestartet");

/* ============================================================
   A1: DATENGENERIERUNG
   ============================================================ */

function groundTruth(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateXValues(N = 100) {
    const xs = [];
    for (let i = 0; i < N; i++) {
        xs.push(Math.random() * 4 - 2);
    }
    return xs;
}

function computeYValues(xs) {
    return xs.map(x => groundTruth(x));
}

function gaussianNoise(mean = 0, variance = 0.05) {
    const u1 = Math.random();
    const u2 = Math.random();
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + Math.sqrt(variance) * randStdNormal;
}

function addNoise(ys, variance = 0.05) {
    return ys.map(y => y + gaussianNoise(0, variance));
}

function splitTrainTest(xs, ys) {
    const indices = [...xs.keys()];
    indices.sort(() => Math.random() - 0.5);

    const trainIdx = indices.slice(0, 50);
    const testIdx = indices.slice(50);

    return {
        train: {
            x: trainIdx.map(i => xs[i]),
            y: trainIdx.map(i => ys[i])
        },
        test: {
            x: testIdx.map(i => xs[i]),
            y: testIdx.map(i => ys[i])
        }
    };
}

/* ===============================
   DATEN ERZEUGEN
================================ */

const xs = generateXValues(100);
const ys_clean = computeYValues(xs);
const { train, test } = splitTrainTest(xs, ys_clean);

const ys_train_noisy = addNoise(train.y);
const ys_test_noisy = addNoise(test.y);

/* ============================================================
   PLOTTING A1
   ============================================================ */

function plotDatasetClean(train, test) {
    Plotly.newPlot('plot-clean', [
        { x: train.x, y: train.y, mode: 'markers', name: 'Train', marker: { color: 'blue' }},
        { x: test.x, y: test.y, mode: 'markers', name: 'Test', marker: { color: 'red' }}
    ], { title: 'Unverrauschte Daten' });
}

function plotDatasetNoisy(train, test, noisyTrain, noisyTest) {
    Plotly.newPlot('plot-noisy', [
        { x: train.x, y: noisyTrain, mode: 'markers', name: 'Train', marker: { color: 'green' }},
        { x: test.x, y: noisyTest, mode: 'markers', name: 'Test', marker: { color: 'orange' }}
    ], { title: 'Verrauschte Daten' });
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
        xsTensor: tf.tensor2d(xs, [xs.length, 1]),
        ysTensor: tf.tensor2d(ys, [ys.length, 1])
    };
}

/* ============================================================
   TRAININGS
   ============================================================ */

async function trainCleanModel() {
    const model = createModel();
    const t = toTensors(train.x, train.y);

    await model.fit(t.xsTensor, t.ysTensor, { epochs: 200, batchSize: 32, verbose: 0 });

    return {
        model,
        trainLoss: model.evaluate(t.xsTensor, t.ysTensor).dataSync()[0]
    };
}

async function trainBestFitModel() {
    const model = createModel();
    const t = toTensors(train.x, ys_train_noisy);

    await model.fit(t.xsTensor, t.ysTensor, { epochs: 150, batchSize: 32, verbose: 0 });

    return {
        model,
        trainLoss: model.evaluate(t.xsTensor, t.ysTensor).dataSync()[0]
    };
}

async function trainOverfitModel() {
    const model = createModel();
    const t = toTensors(train.x, ys_train_noisy);

    await model.fit(t.xsTensor, t.ysTensor, { epochs: 1500, batchSize: 32, verbose: 0 });

    return {
        model,
        trainLoss: model.evaluate(t.xsTensor, t.ysTensor).dataSync()[0]
    };
}

/* ============================================================
   PREDICTION
   ============================================================ */

function predictCurve(model) {
    const xs = [];
    for (let x = -2; x <= 2; x += 0.01) xs.push(x);

    const ys = model.predict(tf.tensor2d(xs, [xs.length, 1])).dataSync();

    return { xs, ys };
}

/* ============================================================
   PLOTS A2–A4
   ============================================================ */

function plotModel(id, dataY, model) {
    const curve = predictCurve(model);

    Plotly.newPlot(id, [
        { x: train.x, y: dataY, mode: 'markers' },
        { x: curve.xs, y: curve.ys, mode: 'lines' }
    ]);
}

/* ============================================================
   LAZY LOADING
   ============================================================ */

let loaded = { a1:false, a2:false, a3:false, a4:false };

async function loadA1() {
    if (loaded.a1) return;
    plotDatasetClean(train, test);
    plotDatasetNoisy(train, test, ys_train_noisy, ys_test_noisy);
    loaded.a1 = true;
}

async function loadA2() {
    if (loaded.a2) return;
    const res = await trainCleanModel();
    plotModel("plot-clean-train", train.y, res.model);
    loaded.a2 = true;
}

async function loadA3() {
    if (loaded.a3) return;
    const res = await trainBestFitModel();
    plotModel("plot-best-train", ys_train_noisy, res.model);
    loaded.a3 = true;
}

async function loadA4() {
    if (loaded.a4) return;
    const res = await trainOverfitModel();
    plotModel("plot-overfit-train", ys_train_noisy, res.model);
    loaded.a4 = true;
}

/* ============================================================
   UI LOGIK
   ============================================================ */

function showSection(id) {
    document.querySelectorAll('.content-section')
        .forEach(sec => sec.style.display = (sec.id === id ? 'block' : 'none'));
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

            await new Promise(r => setTimeout(r, 50));

            if (index === 0) await loadA1();
            if (index === 1) await loadA2();
            if (index === 2) await loadA3();
            if (index === 3) await loadA4();

            if (loading) loading.style.display = "none";
        }
    });
});
