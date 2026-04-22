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
        train: { x: trainIdx.map(i => xs[i]), y: trainIdx.map(i => ys[i]) },
        test: { x: testIdx.map(i => xs[i]), y: testIdx.map(i => ys[i]) }
    };
}

// Daten initial erzeugen
const xs_raw = generateXValues(100);
const ys_clean_raw = computeYValues(xs_raw);
const { train, test } = splitTrainTest(xs_raw, ys_clean_raw);

const ys_train_noisy = addNoise(train.y);
const ys_test_noisy = addNoise(test.y);

/* ============================================================
   MODELL-LOGIK
   ============================================================ */

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    return model;
}

function toTensors(xs, ys) {
    return {
        xsTensor: tf.tensor2d(xs, [xs.length, 1]),
        ysTensor: tf.tensor2d(ys, [ys.length, 1])
    };
}

function predictCurve(model) {
    const xs = [];
    for (let x = -2; x <= 2; x += 0.05) xs.push(x);
    const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
    const ysTensor = model.predict(xsTensor);
    return { xs, ys: Array.from(ysTensor.dataSync()) };
}

function hideLoader(id) {
    const loader = document.getElementById(id);
    if (loader) loader.style.display = "none";
}

/* ============================================================
   TRAININGS-FUNKTIONEN
   ============================================================ */

async function trainCleanModel(train, test) {
    const model = createModel();
    const t = toTensors(train.x, train.y);
    await model.fit(t.xsTensor, t.ysTensor, { epochs: 200, verbose: 0 });
    const trainLoss = model.evaluate(t.xsTensor, t.ysTensor).dataSync()[0];
    const testT = toTensors(test.x, test.y);
    const testLoss = model.evaluate(testT.xsTensor, testT.ysTensor).dataSync()[0];
    return { model, trainLoss, testLoss };
}

async function trainBestFitModel(train, test, noisyTrainY, noisyTestY) {
    const model = createModel();
    const t = toTensors(train.x, noisyTrainY);
    await model.fit(t.xsTensor, t.ysTensor, { epochs: 150, verbose: 0 });
    const trainLoss = model.evaluate(t.xsTensor, t.ysTensor).dataSync()[0];
    const testT = toTensors(test.x, noisyTestY);
    const testLoss = model.evaluate(testT.xsTensor, testT.ysTensor).dataSync()[0];
    return { model, trainLoss, testLoss };
}

async function trainOverfitModel(train, test, noisyTrainY, noisyTestY) {
    const model = createModel();
    const t = toTensors(train.x, noisyTrainY);
    await model.fit(t.xsTensor, t.ysTensor, { epochs: 1500, verbose: 0 });
    const trainLoss = model.evaluate(t.xsTensor, t.ysTensor).dataSync()[0];
    const testT = toTensors(test.x, noisyTestY);
    const testLoss = model.evaluate(testT.xsTensor, testT.ysTensor).dataSync()[0];
    return { model, trainLoss, testLoss };
}

/* ============================================================
   VISUALISIERUNG
   ============================================================ */

function plotResults(containerTrain, containerTest, lossContainer, trainData, testData, model, trainL, testL, titleSuffix) {
    const curve = predictCurve(model);
    
    const traceTrain = { x: trainData.x, y: trainData.y, mode: 'markers', name: 'Train', marker: { color: 'blue' } };
    const traceTest = { x: testData.x, y: testData.y, mode: 'markers', name: 'Test', marker: { color: 'red' } };
    const traceCurve = { x: curve.xs, y: curve.ys, mode: 'lines', name: 'Modell', line: { color: 'black', width: 3 } };

    Plotly.newPlot(containerTrain, [traceTrain, traceCurve], { title: `Training (${titleSuffix})` });
    Plotly.newPlot(containerTest, [traceTest, traceCurve], { title: `Test (${titleSuffix})` });

    document.getElementById(lossContainer).innerHTML = `
        <div class="content-box" style="background: #eee; text-align: center;">
            <strong>Train Loss:</strong> ${trainL.toFixed(6)} | <strong>Test Loss:</strong> ${testL.toFixed(6)}
        </div>
    `;
}

// Initialer Start der Prozesse
// A1 & R1
function initA1() {
    Plotly.newPlot('plot-clean', [
        { x: train.x, y: train.y, mode: 'markers', name: 'Train' },
        { x: test.x, y: test.y, mode: 'markers', name: 'Test' }
    ], { title: 'Clean Data' });
    
    Plotly.newPlot('plot-noisy', [
        { x: train.x, y: ys_train_noisy, mode: 'markers', name: 'Train Noisy' },
        { x: test.x, y: ys_test_noisy, mode: 'markers', name: 'Test Noisy' }
    ], { title: 'Noisy Data' });
    hideLoader('loader-a1');
}

initA1();

// A2
trainCleanModel(train, test).then(res => {
    hideLoader('loader-clean');
    plotResults('plot-clean-train', 'plot-clean-test', 'loss-clean', train, test, res.model, res.trainLoss, res.testLoss, "Clean");
});

// A3
trainBestFitModel(train, test, ys_train_noisy, ys_test_noisy).then(res => {
    hideLoader('loader-best');
    plotResults('plot-best-train', 'plot-best-test', 'loss-best', {x: train.x, y: ys_train_noisy}, {x: test.x, y: ys_test_noisy}, res.model, res.trainLoss, res.testLoss, "Best-Fit");
});

// A4
trainOverfitModel(train, test, ys_train_noisy, ys_test_noisy).then(res => {
    hideLoader('loader-overfit');
    plotResults('plot-overfit-train', 'plot-overfit-test', 'loss-overfit', {x: train.x, y: ys_train_noisy}, {x: test.x, y: ys_test_noisy}, res.model, res.trainLoss, res.testLoss, "Overfit");
});

/* ============================================================
   UI LOGIK
   ============================================================ */

function showSection(id) {
    document.querySelectorAll('.content-section').forEach(sec => sec.style.display = (sec.id === id) ? 'block' : 'none');
}

document.querySelectorAll(".accordion-header").forEach(header => {
    header.addEventListener("click", () => {
        const item = header.parentElement;
        item.classList.toggle("active");
    });
});
