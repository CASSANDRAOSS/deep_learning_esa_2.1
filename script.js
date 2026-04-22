console.log("TFJS Regression Projekt gestartet");

/* ============================================================
   A1: DATENGENERIERUNG
   ------------------------------------------------------------
   - Erzeugt 100 gleichverteilte x-Werte im Intervall [-2, 2]
   - Berechnet die Ground-Truth y(x)
   - Fügt optional Gaussian Noise hinzu
   - Splittet in 50 Trainings- und 50 Testdaten
   ============================================================ */

// Ground-Truth Funktion y(x)
function groundTruth(x) {
    // die Funktion aus der Aufgabenstellung
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

// 100 gleichverteilte x-Werte in [-2, 2]
function generateXValues(N = 100) {
    const xs = [];
    for (let i = 0; i < N; i++) {
        xs.push(Math.random() * 4 - 2); // [-2, 2]
    }
    return xs;
}

// y-Werte berechnen (unverrauscht)
function computeYValues(xs) {
    return xs.map(x => groundTruth(x));
}

// Gaussian Noise (Varianz = 0.05)
function gaussianNoise(mean = 0, variance = 0.05) {
    // erzeugt normalverteilte Zufallszahlen
    const u1 = Math.random();
    const u2 = Math.random();
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + Math.sqrt(variance) * randStdNormal;
}

// Rauschen zu y hinzufügen
function addNoise(ys, variance = 0.05) {
    return ys.map(y => y + gaussianNoise(0, variance));
}

// Zufälliger Split in 50 Train / 50 Test
function splitTrainTest(xs, ys) {
    const indices = [...xs.keys()];
    indices.sort(() => Math.random() - 0.5);

    const trainIdx = indices.slice(0, 50);
    const testIdx = indices.slice(50);

    const train = {
        x: trainIdx.map(i => xs[i]),
        y: trainIdx.map(i => ys[i])
    };

    const test = {
        x: testIdx.map(i => xs[i]),
        y: testIdx.map(i => ys[i])
    };

    return { train, test };
}

// ===============================
// DATEN ERZEUGEN
// ===============================

const xs = generateXValues(100);
const ys_clean = computeYValues(xs);

const { train, test } = splitTrainTest(xs, ys_clean);

// verrauschte Versionen
const ys_train_noisy = addNoise(train.y);
const ys_test_noisy = addNoise(test.y);

// Debug-Ausgabe
console.log("Train clean:", train);
console.log("Test clean:", test);
console.log("Train noisy:", ys_train_noisy);
console.log("Test noisy:", ys_test_noisy);

/* ------------------------------------------------------------
   Visualisiert die unverrauschten und verrauschten Daten
   - Train/Test farblich getrennt
   - Plotly für interaktive Diagramme
   ------------------------------------------------------------ */

function plotDatasetClean(train, test) {
    const traceTrain = {
        x: train.x,
        y: train.y,
        mode: 'markers',
        name: 'Train (clean)',
        marker: { color: 'blue' }
    };

    const traceTest = {
        x: test.x,
        y: test.y,
        mode: 'markers',
        name: 'Test (clean)',
        marker: { color: 'red' }
    };

    const layout = {
        title: 'Unverrauschte Daten',
        xaxis: { title: 'x', automargin: true },
        yaxis: { title: 'y', automargin: true },
        margin: { l: 50, r: 30, t: 50, b: 50 },
        width: 400  
    };

    Plotly.newPlot('plot-clean', [traceTrain, traceTest], layout);
}

function plotDatasetNoisy(train, test, noisyTrain, noisyTest) {
    const traceTrain = {
        x: train.x,
        y: noisyTrain,
        mode: 'markers',
        name: 'Train (noisy)',
        marker: { color: 'green' }
    };

    const traceTest = {
        x: test.x,
        y: noisyTest,
        mode: 'markers',
        name: 'Test (noisy)',
        marker: { color: 'orange' }
    };

    const layout = {
        title: 'Verrauschte Daten',
        xaxis: { title: 'x', automargin: true },
        yaxis: { title: 'y', automargin: true },
        margin: { l: 50, r: 30, t: 50, b: 50 },
        width: 400
    };

    Plotly.newPlot('plot-noisy', [traceTrain, traceTest], layout);
}

// Diagramme zeichnen
plotDatasetClean(train, test);
plotDatasetNoisy(train, test, ys_train_noisy, ys_test_noisy);


/* ============================================================
   A2: MODELL FÜR UNVERRAUCHTE DATEN
   ------------------------------------------------------------
   - 2 Hidden Layers mit je 100 Neuronen
   - ReLU Aktivierung
   - Linearer Output für Regression
   - Adam Optimizer (LR=0.01)
   ============================================================ */

function createModel() {
    const model = tf.sequential();

    // Hidden Layer 1
    model.add(tf.layers.dense({
        units: 100,
        activation: 'relu',
        inputShape: [1]
    }));

    // Hidden Layer 2
    model.add(tf.layers.dense({
        units: 100,
        activation: 'relu'
    }));

    // Output Layer (linear)
    model.add(tf.layers.dense({
        units: 1,
        activation: 'linear'
    }));

    // Compile
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    return model;
}

// ===============================
// A3: MODELL FÜR VERRAUCHTE DATEN (BEST-FIT)
// ===============================


async function trainBestFitModel(train, test, noisyTrainY, noisyTestY) {
    const model = createModel();

    const trainTensors = toTensors(train.x, noisyTrainY);
    const testTensors = toTensors(test.x, noisyTestY);

    console.log("Training Best-Fit Modell (noisy)...");

    // moderate Anzahl an Epochen → gute Generalisierung
    await model.fit(trainTensors.xsTensor, trainTensors.ysTensor, {
        epochs: 150,
        batchSize: 32,
        shuffle: true,
        verbose: 0
    });

    const trainLoss = model.evaluate(trainTensors.xsTensor, trainTensors.ysTensor).dataSync()[0];
    const testLoss = model.evaluate(testTensors.xsTensor, testTensors.ysTensor).dataSync()[0];

    console.log("Best-Fit Train Loss:", trainLoss);
    console.log("Best-Fit Test Loss:", testLoss);

    return { model, trainLoss, testLoss };
}

// ===============================
// A4: OVERFIT-MODELL (lange trainieren)
// ===============================

async function trainOverfitModel(train, test, noisyTrainY, noisyTestY) {
    const model = createModel();

    const trainTensors = toTensors(train.x, noisyTrainY);
    const testTensors = toTensors(test.x, noisyTestY);

    console.log("Training Overfit Modell (noisy)...");

    // ABSICHTLICH VIEL ZU VIELE EPOCHEN
    await model.fit(trainTensors.xsTensor, trainTensors.ysTensor, {
        epochs: 1500,   // Overfitting garantiert
        batchSize: 32,
        shuffle: true,
        verbose: 0
    });

    const trainLoss = model.evaluate(trainTensors.xsTensor, trainTensors.ysTensor).dataSync()[0];
    const testLoss = model.evaluate(testTensors.xsTensor, testTensors.ysTensor).dataSync()[0];

    console.log("Overfit Train Loss:", trainLoss);
    console.log("Overfit Test Loss:", testLoss);

    return { model, trainLoss, testLoss };
}


// Wandelt JS-Arrays in TensorFlow-Tensoren um
function toTensors(xs, ys) {
    const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
    const ysTensor = tf.tensor2d(ys, [ys.length, 1]);
    return { xsTensor, ysTensor };
}


/* ------------------------------------------------------------
   Trainiert ein Modell auf den unverrauschten Daten
   - 200 Epochen → kein Overfitting möglich
   ------------------------------------------------------------ */
async function trainCleanModel(train, test) {
    const model = createModel();

    const trainTensors = toTensors(train.x, train.y);
    const testTensors = toTensors(test.x, test.y);

    console.log("Training Modell (clean)...");

    await model.fit(trainTensors.xsTensor, trainTensors.ysTensor, {
        epochs: 200,
        batchSize: 32,
        shuffle: true,
        verbose: 0
    });

    // Loss berechnen
    const trainLoss = model.evaluate(trainTensors.xsTensor, trainTensors.ysTensor).dataSync()[0];
    const testLoss = model.evaluate(testTensors.xsTensor, testTensors.ysTensor).dataSync()[0];

    console.log("Clean Model Train Loss:", trainLoss);
    console.log("Clean Model Test Loss:", testLoss);

    return { model, trainLoss, testLoss };
}


/* ------------------------------------------------------------
   Erzeugt eine glatte Kurve für die Modellvorhersage
   - 401 Punkte im Bereich [-2, 2]
   ------------------------------------------------------------ */
function predictCurve(model) {
    const xs = [];
    for (let x = -2; x <= 2; x += 0.01) {
        xs.push(x);
    }

    const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
    const ysTensor = model.predict(xsTensor);
    const ys = Array.from(ysTensor.dataSync());

    return { xs, ys };
}

/* ------------------------------------------------------------
   Visualisiert Predictions des Clean-Modells
   - Punkte = Trainings/Testdaten
   - Linie = Modellvorhersage
   - Loss-Werte werden unterhalb angezeigt
   ------------------------------------------------------------ */

function plotCleanPredictions(train, test, model, trainLoss, testLoss) {
    const curve = predictCurve(model);

    // TRAIN-PLOT
    const traceTrainPoints = {
        x: train.x,
        y: train.y,
        mode: 'markers',
        name: 'Train (clean)',
        marker: { color: 'blue' }
    };

    const traceTrainCurve = {
        x: curve.xs,
        y: curve.ys,
        mode: 'lines',
        name: 'Prediction',
        line: { color: 'black' }
    };

    Plotly.newPlot('plot-clean-train', [traceTrainPoints, traceTrainCurve], {
        title: 'Vorhersage auf Trainingsdaten (clean)',
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });

    // TEST-PLOT
    const traceTestPoints = {
        x: test.x,
        y: test.y,
        mode: 'markers',
        name: 'Test (clean)',
        marker: { color: 'red' }
    };

    const traceTestCurve = {
        x: curve.xs,
        y: curve.ys,
        mode: 'lines',
        name: 'Prediction',
        line: { color: 'black' }
    };

    Plotly.newPlot('plot-clean-test', [traceTestPoints, traceTestCurve], {
        title: 'Vorhersage auf Testdaten (clean)',
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });

    // LOSS-WERTE UNTER DEN PLOTS
    document.getElementById("loss-clean").innerHTML = `
        <p><b>Train Loss:</b> ${trainLoss.toFixed(6)}</p>
        <p><b>Test Loss:</b> ${testLoss.toFixed(6)}</p>
    `;
}

function plotBestFitPredictions(train, test, model, trainLoss, testLoss, noisyTrainY, noisyTestY) {
    const curve = predictCurve(model);

    // TRAIN-PLOT
    const traceTrainPoints = {
        x: train.x,
        y: noisyTrainY,
        mode: 'markers',
        name: 'Train (noisy)',
        marker: { color: 'blue' }
    };

    const traceTrainCurve = {
        x: curve.xs,
        y: curve.ys,
        mode: 'lines',
        name: 'Prediction',
        line: { color: 'black' }
    };

    Plotly.newPlot('plot-best-train', [traceTrainPoints, traceTrainCurve], {
        title: 'Best-Fit Vorhersage (Train, noisy)',
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });

    // TEST-PLOT
    const traceTestPoints = {
        x: test.x,
        y: noisyTestY,
        mode: 'markers',
        name: 'Test (noisy)',
        marker: { color: 'red' }
    };

    const traceTestCurve = {
        x: curve.xs,
        y: curve.ys,
        mode: 'lines',
        name: 'Prediction',
        line: { color: 'black' }
    };

    Plotly.newPlot('plot-best-test', [traceTestPoints, traceTestCurve], {
        title: 'Best-Fit Vorhersage (Test, noisy)',
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });

    // LOSS-WERTE
    document.getElementById("loss-best").innerHTML = `
        <p><b>Train Loss:</b> ${trainLoss.toFixed(6)}</p>
        <p><b>Test Loss:</b> ${testLoss.toFixed(6)}</p>
    `;
}


function plotOverfitPredictions(train, test, model, trainLoss, testLoss, noisyTrainY, noisyTestY) {
    const curve = predictCurve(model);

    // TRAIN-PLOT
    const traceTrainPoints = {
        x: train.x,
        y: noisyTrainY,
        mode: 'markers',
        name: 'Train (noisy)',
        marker: { color: 'blue' }
    };

    const traceTrainCurve = {
        x: curve.xs,
        y: curve.ys,
        mode: 'lines',
        name: 'Prediction',
        line: { color: 'black' }
    };

    Plotly.newPlot('plot-overfit-train', [traceTrainPoints, traceTrainCurve], {
        title: 'Overfit Vorhersage (Train, noisy)',
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });

    // TEST-PLOT
    const traceTestPoints = {
        x: test.x,
        y: noisyTestY,
        mode: 'markers',
        name: 'Test (noisy)',
        marker: { color: 'red' }
    };

    const traceTestCurve = {
        x: curve.xs,
        y: curve.ys,
        mode: 'lines',
        name: 'Prediction',
        line: { color: 'black' }
    };

    Plotly.newPlot('plot-overfit-test', [traceTestPoints, traceTestCurve], {
        title: 'Overfit Vorhersage (Test, noisy)',
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });

    // LOSS-WERTE
    document.getElementById("loss-overfit").innerHTML = `
        <p><b>Train Loss:</b> ${trainLoss.toFixed(6)}</p>
        <p><b>Test Loss:</b> ${testLoss.toFixed(6)}</p>
    `;
}


trainCleanModel(train, test).then(result => {
    plotCleanPredictions(train, test, result.model, result.trainLoss, result.testLoss);
});

trainBestFitModel(train, test, ys_train_noisy, ys_test_noisy).then(result => {
    plotBestFitPredictions(
        train,
        test,
        result.model,
        result.trainLoss,
        result.testLoss,
        ys_train_noisy,
        ys_test_noisy
    );
});

trainOverfitModel(train, test, ys_train_noisy, ys_test_noisy).then(result => {
    plotOverfitPredictions(
        train,
        test,
        result.model,
        result.trainLoss,
        result.testLoss,
        ys_train_noisy,
        ys_test_noisy
    );
});


// ===============================
// Akkordeon-Logik
// ===============================
/* ------------------------------------------------------------
   Navigation zwischen den Seitenabschnitten
   ------------------------------------------------------------ */


function showSection(id) {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(sec => {
        sec.style.display = (sec.id === id) ? 'block' : 'none';
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Akkordeon-Logik (für Modelle + Doku)
document.querySelectorAll(".accordion-header").forEach(header => {
    header.addEventListener("click", () => {
        const item = header.parentElement;

        // Nur innerhalb des jeweiligen Akkordeons schließen
        const parentAccordion = item.parentElement;
        parentAccordion.querySelectorAll(".accordion-item").forEach(other => {
            if (other !== item) other.classList.remove("active");
        });

        item.classList.toggle("active");
    });
}
);



function scrollToSection(id) {
    const el = document.getElementById(id);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
    }
}


function showAccordion() {
    const acc = document.getElementById("accordion");
    acc.style.display = "block";
    acc.scrollIntoView({ behavior: "smooth" });
}




