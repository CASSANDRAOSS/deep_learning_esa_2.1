console.log("TFJS Regression Projekt gestartet");

/* ============================================================
   DATENGENERIERUNG & HELFER
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

// Global verfügbare Daten
const xs_raw = generateXValues(100);
const ys_clean_raw = computeYValues(xs_raw);
const { train, test } = splitTrainTest(xs_raw, ys_clean_raw);
const ys_train_noisy = addNoise(train.y);
const ys_test_noisy = addNoise(test.y);

/* ============================================================
   TENSORFLOW LOGIK
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
    for (let x = -2.1; x <= 2.1; x += 0.05) xs.push(x);
    const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
    const ysTensor = model.predict(xsTensor);
    const ys = Array.from(ysTensor.dataSync());
    // Aufräumen um Speicherlecks zu verhindern
    xsTensor.dispose();
    ysTensor.dispose();
    return { xs, ys };
}

/* ============================================================
   UI FUNKTIONEN (Laden & Plotten)
   ============================================================ */

function hideLoader(id) {
    const loader = document.getElementById(id);
    if (loader) {
        console.log("Entferne Loader:", id);
        loader.style.display = "none"; // Sicher wegschalten
        loader.remove(); // Und komplett aus dem DOM löschen
    }
}

function plotResults(containerTrain, containerTest, lossContainer, trainData, testData, model, trainL, testL, titleSuffix) {
    try {
        const curve = predictCurve(model);
        
        const traceTrain = { x: trainData.x, y: trainData.y, mode: 'markers', name: 'Daten (Train)', marker: { color: 'blue' } };
        const traceTest = { x: testData.x, y: testData.y, mode: 'markers', name: 'Daten (Test)', marker: { color: 'red' } };
        const traceCurve = { x: curve.xs, y: curve.ys, mode: 'lines', name: 'Modell Kurve', line: { color: 'black', width: 3 } };

        const layout = { margin: { t: 40, b: 40, l: 40, r: 20 }, hovermode: 'closest' };

        Plotly.newPlot(containerTrain, [traceTrain, traceCurve], { ...layout, title: `Training (${titleSuffix})` });
        Plotly.newPlot(containerTest, [traceTest, traceCurve], { ...layout, title: `Test (${titleSuffix})` });

        const lossBox = document.getElementById(lossContainer);
        if (lossBox) {
            lossBox.innerHTML = `
                <div class="content-box" style="background: #efefef; text-align: center; border: 1px solid #ccc;">
                    <strong>Ergebnis ${titleSuffix}:</strong><br>
                    Train Loss: <code>${trainL.toFixed(6)}</code> | Test Loss: <code>${testL.toFixed(6)}</code>
                </div>
            `;
        }
    } catch (err) {
        console.error("Fehler beim Plotten von " + titleSuffix, err);
    }
}

/* ============================================================
   TRAININGS-ABLAUF (Async)
   ============================================================ */

async function runTrainings() {
    // A1: Datengenerierung visualisieren (kein Training nötig)
    Plotly.newPlot('plot-clean', [{ x: train.x, y: train.y, mode: 'markers', name: 'Train' }, { x: test.x, y: test.y, mode: 'markers', name: 'Test' }], { title: 'Unverrauschte Daten' });
    Plotly.newPlot('plot-noisy', [{ x: train.x, y: ys_train_noisy, mode: 'markers', name: 'Train Noisy' }, { x: test.x, y: ys_test_noisy, mode: 'markers', name: 'Test Noisy' }], { title: 'Verrauschte Daten' });
    hideLoader('loader-a1');

    // A2: Clean Model
    try {
        const model2 = createModel();
        const t = toTensors(train.x, train.y);
        const testT = toTensors(test.x, test.y);
        await model2.fit(t.xsTensor, t.ysTensor, { epochs: 200, verbose: 0 });
        const trL = model2.evaluate(t.xsTensor, t.ysTensor).dataSync()[0];
        const teL = model2.evaluate(testT.xsTensor, testT.ysTensor).dataSync()[0];
        
        hideLoader('loader-clean');
        plotResults('plot-clean-train', 'plot-clean-test', 'loss-clean', train, test, model2, trL, teL, "Clean");
    } catch (e) { console.error(e); }

    // A3: Best-Fit (moderate Epochen)
    try {
        const model3 = createModel();
        const t = toTensors(train.x, ys_train_noisy);
        const testT = toTensors(test.x, ys_test_noisy);
        await model3.fit(t.xsTensor, t.ysTensor, { epochs: 150, verbose: 0 });
        const trL = model3.evaluate(t.xsTensor, t.ysTensor).dataSync()[0];
        const teL = model3.evaluate(testT.xsTensor, testT.ysTensor).dataSync()[0];

        hideLoader('loader-best');
        plotResults('plot-best-train', 'plot-best-test', 'loss-best', {x: train.x, y: ys_train_noisy}, {x: test.x, y: ys_test_noisy}, model3, trL, teL, "Best-Fit");
    } catch (e) { console.error(e); }

    // A4: Overfit (viele Epochen)
    try {
        const model4 = createModel();
        const t = toTensors(train.x, ys_train_noisy);
        const testT = toTensors(test.x, ys_test_noisy);
        await model4.fit(t.xsTensor, t.ysTensor, { epochs: 1500, verbose: 0 });
        const trL = model4.evaluate(t.xsTensor, t.ysTensor).dataSync()[0];
        const teL = model4.evaluate(testT.xsTensor, testT.ysTensor).dataSync()[0];

        hideLoader('loader-overfit');
        plotResults('plot-overfit-train', 'plot-overfit-test', 'loss-overfit', {x: train.x, y: ys_train_noisy}, {x: test.x, y: ys_test_noisy}, model4, trL, teL, "Overfit");
    } catch (e) { console.error(e); }
}

// Start der Kette
runTrainings();

/* ============================================================
   NAV & AKKORDEON
   ============================================================ */

function showSection(id) {
    document.querySelectorAll('.content-section').forEach(sec => {
        sec.style.display = (sec.id === id) ? 'block' : 'none';
    });
}

document.querySelectorAll(".accordion-header").forEach(header => {
    header.addEventListener("click", () => {
        header.parentElement.classList.toggle("active");
    });
});
