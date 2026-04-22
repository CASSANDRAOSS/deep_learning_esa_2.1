console.log("TFJS Regression Projekt gestartet");

/* ============================================================
   🔹 LOADING HANDLING
   ============================================================ */

function showLoading(id) {
    const el = document.getElementById(id);
    if (el) {
        el.innerHTML = "<p style='text-align:center;'>lädt, bitte warten...</p>";
    }
}

function hideLoading(id) {
    const el = document.getElementById(id);
    if (el) {
        el.innerHTML = "";
    }
}


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


/* ============================================================
   DATEN ERZEUGEN
   ============================================================ */

const xs = generateXValues(100);
const ys_clean = computeYValues(xs);

const { train, test } = splitTrainTest(xs, ys_clean);

const ys_train_noisy = addNoise(train.y);
const ys_test_noisy = addNoise(test.y);


/* ============================================================
   PLOTS (MIT LOADING)
   ============================================================ */

function plotDatasetClean(train, test) {
    showLoading("plot-clean");

    setTimeout(() => {
        Plotly.newPlot('plot-clean', [
            { x: train.x, y: train.y, mode: 'markers', name: 'Train', marker: { color: 'blue' } },
            { x: test.x, y: test.y, mode: 'markers', name: 'Test', marker: { color: 'red' } }
        ]);

        hideLoading("plot-clean");
    }, 50);
}

function plotDatasetNoisy(train, test, noisyTrain, noisyTest) {
    showLoading("plot-noisy");

    setTimeout(() => {
        Plotly.newPlot('plot-noisy', [
            { x: train.x, y: noisyTrain, mode: 'markers', name: 'Train', marker: { color: 'green' } },
            { x: test.x, y: noisyTest, mode: 'markers', name: 'Test', marker: { color: 'orange' } }
        ]);

        hideLoading("plot-noisy");
    }, 50);
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
   TRAINING
   ============================================================ */

async function trainModel(train, test, trainY, testY, epochs) {
    const model = createModel();

    const trainT = toTensors(train.x, trainY);
    const testT = toTensors(test.x, testY);

    await model.fit(trainT.xsTensor, trainT.ysTensor, {
        epochs,
        batchSize: 32,
        shuffle: true,
        verbose: 0
    });

    const trainLoss = model.evaluate(trainT.xsTensor, trainT.ysTensor).dataSync()[0];
    const testLoss = model.evaluate(testT.xsTensor, testT.ysTensor).dataSync()[0];

    return { model, trainLoss, testLoss };
}


/* ============================================================
   PREDICTION
   ============================================================ */

function predictCurve(model) {
    const xs = [];
    for (let x = -2; x <= 2; x += 0.01) xs.push(x);

    const ys = Array.from(model.predict(tf.tensor2d(xs, [xs.length, 1])).dataSync());
    return { xs, ys };
}


/* ============================================================
   PLOT FUNCTIONS (MIT LOADING)
   ============================================================ */

function plotResult(plotId, lossId, train, test, model, trainLoss, testLoss, trainY, testY) {
    showLoading(plotId);

    setTimeout(() => {
        const curve = predictCurve(model);

        Plotly.newPlot(plotId, [
            { x: train.x, y: trainY, mode: 'markers', name: 'Train' },
            { x: test.x, y: testY, mode: 'markers', name: 'Test' },
            { x: curve.xs, y: curve.ys, mode: 'lines', name: 'Prediction' }
        ]);

        document.getElementById(lossId).innerHTML = `
            <p><b>Train Loss:</b> ${trainLoss.toFixed(6)}</p>
            <p><b>Test Loss:</b> ${testLoss.toFixed(6)}</p>
        `;

        hideLoading(plotId);
    }, 50);
}


/* ============================================================
   EXECUTION
   ============================================================ */

plotDatasetClean(train, test);
plotDatasetNoisy(train, test, ys_train_noisy, ys_test_noisy);

trainModel(train, test, train.y, test.y, 200)
    .then(r => plotResult("plot-clean-train", "loss-clean", train, test, r.model, r.trainLoss, r.testLoss, train.y, test.y));

trainModel(train, test, ys_train_noisy, ys_test_noisy, 150)
    .then(r => plotResult("plot-best-train", "loss-best", train, test, r.model, r.trainLoss, r.testLoss, ys_train_noisy, ys_test_noisy));

trainModel(train, test, ys_train_noisy, ys_test_noisy, 1500)
    .then(r => plotResult("plot-overfit-train", "loss-overfit", train, test, r.model, r.trainLoss, r.testLoss, ys_train_noisy, ys_test_noisy));


/* ============================================================
   UI
   ============================================================ */

function showSection(id) {
    document.querySelectorAll('.content-section').forEach(sec => {
        sec.style.display = (sec.id === id) ? 'block' : 'none';
    });
}

document.querySelectorAll(".accordion-header").forEach(header => {
    header.addEventListener("click", () => {
        const item = header.parentElement;

        item.parentElement.querySelectorAll(".accordion-item").forEach(other => {
            if (other !== item) other.classList.remove("active");
        });

        item.classList.toggle("active");
    });
});
