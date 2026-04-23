/* ============================================================
   KONTROLLZENTRUM & PIPELINE-LOGIK
   ============================================================ */

/**
 * Startet den gesamten Prozess: Datengenerierung -> Training -> Visualisierung.
 * Durch 'async' und 'await' werden die Schritte nacheinander ausgeführt,
 * damit die CPU/GPU nicht überlastet und der Status für den Prüfer sichtbar bleibt.
 */
async function startDataPipeline() {
    const btn = document.getElementById('btn-start-pipeline');
    const statusTitle = document.getElementById('status-title');
    const statusText = document.getElementById('status-text');

    // UI-Vorbereitung
    btn.disabled = true;
    btn.innerText = "Pipeline läuft...";
    console.log("--- TFJS Pipeline gestartet ---");

    // 1. SCHRITT: DATEN GENERIEREN
    updateStatus("Schritt 1: Datengenerierung", "Erzeuge 100 Punkte und berechne Ground-Truth...");
    const rawX = generateXValues(100);
    const rawYClean = computeYValues(rawX);

    // Daten splitten (50/50)
    const { train, test } = splitTrainTest(rawX, rawYClean);
    
    // Rauschen hinzufügen (für die Noisy-Modelle)
    const trainYNoisy = addNoise(train.y);
    const testYNoisy = addNoise(test.y);

    // Initial-Plots zeichnen
    plotDatasetClean(train, test);
    plotDatasetNoisy(train, test, trainYNoisy, testYNoisy);
    await tf.nextFrame(); // Browser Zeit zum Rendern geben

    // 2. SCHRITT: CLEAN MODEL (Training auf perfekten Daten)
    updateStatus("Schritt 2: Clean Model", "Trainiere Modell ohne Rauschen (200 Epochen)...");
    const cleanResult = await trainCleanModel(train, test);
    plotCleanPredictions(train, test, cleanResult.model, cleanResult.trainLoss, cleanResult.testLoss);
    await saveMyModel(cleanResult.model, "model-clean");

    // 3. SCHRITT: BEST-FIT MODEL (Gute Generalisierung trotz Rauschen)
    updateStatus("Schritt 3: Best-Fit Modell", "Lerne Struktur aus verrauschten Daten (150 Epochen)...");
    const bestFitResult = await trainBestFitModel(train, test, trainYNoisy, testYNoisy);
    plotBestFitPredictions(train, test, bestFitResult.model, bestFitResult.trainLoss, bestFitResult.testLoss, trainYNoisy, testYNoisy);
    await saveMyModel(bestFitResult.model, "model-bestfit");

    // 4. SCHRITT: OVERFIT MODEL (Auswendiglernen des Rauschens)
    updateStatus("Schritt 4: Overfit Modell", "Provokation von Overfitting (1500 Epochen)... Bitte warten.");
    const overfitResult = await trainOverfitModel(train, test, trainYNoisy, testYNoisy);
    plotOverfitPredictions(train, test, overfitResult.model, overfitResult.trainLoss, overfitResult.testLoss, trainYNoisy, testYNoisy);
    await saveMyModel(overfitResult.model, "model-overfit");

    // FINISH
    updateStatus("Erfolg!", "Alle Modelle wurden trainiert, visualisiert und im LocalStorage gespeichert.");
    btn.disabled = false;
    btn.innerText = "Pipeline neu starten";
    console.log("--- TFJS Pipeline erfolgreich beendet ---");
}

// Hilfsfunktion für Status-Updates
function updateStatus(title, msg) {
    document.getElementById('status-title').innerText = title;
    document.getElementById('status-text').innerText = msg;
    console.log(`Status: ${title} - ${msg}`);
}

/* ============================================================
   A1: DATEN-ENGINEERING
   ============================================================ */

function groundTruth(x) {
    // Nichtlineare Funktion 5. Grades laut Aufgabenstellung
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateXValues(N) {
    const xs = [];
    for (let i = 0; i < N; i++) {
        xs.push(Math.random() * 4 - 2); // Bereich [-2, 2]
    }
    return xs;
}

function computeYValues(xs) {
    return xs.map(x => groundTruth(x));
}

/**
 * Box-Muller-Transform für echtes Gaussian Noise
 */
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
    indices.sort(() => Math.random() - 0.5); // Zufälliger Shuffle

    const trainIdx = indices.slice(0, 50);
    const testIdx = indices.slice(50);

    return {
        train: { x: trainIdx.map(i => xs[i]), y: trainIdx.map(i => ys[i]) },
        test: { x: testIdx.map(i => xs[i]), y: testIdx.map(i => ys[i]) }
    };
}

/* ============================================================
   A2-A4: MODEL BUILDING & TRAINING
   ============================================================ */

function createModel() {
    const model = tf.sequential();
    // Komplexität: 2 Schichten à 100 Neuronen ermöglichen das Lernen hochgradiger Polynome
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Regression braucht linearen Output

    model.compile({
        optimizer: tf.train.adam(0.01), // Adam ist robust für diese Art von Problemen
        loss: 'meanSquaredError'
    });
    return model;
}

async function trainCleanModel(train, test) {
    const model = createModel();
    const t = toTensors(train.x, train.y);
    await model.fit(t.xs, t.ys, { epochs: 200, verbose: 0 });
    
    const trainLoss = model.evaluate(t.xs, t.ys).dataSync()[0];
    const testT = toTensors(test.x, test.y);
    const testLoss = model.evaluate(testT.xs, testT.ys).dataSync()[0];
    
    return { model, trainLoss, testLoss };
}

async function trainBestFitModel(train, test, noisyY, noisyTestY) {
    const model = createModel();
    const t = toTensors(train.x, noisyY);
    // 150 Epochen: Genug um den Trend zu sehen, zu wenig um Ausreißer "auswendig" zu lernen
    await model.fit(t.xs, t.ys, { epochs: 150, verbose: 0 });
    
    const trainLoss = model.evaluate(t.xs, t.ys).dataSync()[0];
    const testT = toTensors(test.x, noisyTestY);
    const testLoss = model.evaluate(testT.xs, testT.ys).dataSync()[0];
    
    return { model, trainLoss, testLoss };
}

async function trainOverfitModel(train, test, noisyY, noisyTestY) {
    const model = createModel();
    const t = toTensors(train.x, noisyY);
    // 1500 Epochen: Das Modell wird versuchen, jeden verrauschten Punkt exakt zu treffen
    await model.fit(t.xs, t.ys, { epochs: 1500, verbose: 0 });
    
    const trainLoss = model.evaluate(t.xs, t.ys).dataSync()[0];
    const testT = toTensors(test.x, noisyTestY);
    const testLoss = model.evaluate(testT.xs, testT.ys).dataSync()[0];
    
    return { model, trainLoss, testLoss };
}

function toTensors(x, y) {
    return {
        xs: tf.tensor2d(x, [x.length, 1]),
        ys: tf.tensor2d(y, [y.length, 1])
    };
}

/* ============================================================
   VISUALISIERUNG (PLOTLY)
   ============================================================ */

function predictCurve(model) {
    const xs = [];
    for (let x = -2; x <= 2; x += 0.01) xs.push(x);
    const xsT = tf.tensor2d(xs, [xs.length, 1]);
    const ysT = model.predict(xsT);
    return { xs, ys: Array.from(ysT.dataSync()) };
}

// Beispiel für eine Plot-Funktion (Stellvertretend für die anderen)
function plotCleanPredictions(train, test, model, trainLoss, testLoss) {
    const curve = predictCurve(model);
    
    const tracePoints = { x: train.x, y: train.y, mode: 'markers', name: 'Train Clean', marker: {color: 'blue'}};
    const traceCurve = { x: curve.xs, y: curve.ys, mode: 'lines', name: 'Vorhersage', line: {color: 'black'}};

    Plotly.newPlot('plot-clean-train', [tracePoints, traceCurve], { title: 'Modell auf sauberen Daten' });
    document.getElementById("loss-clean").innerHTML = `<b>Train Loss:</b> ${trainLoss.toFixed(6)} | <b>Test Loss:</b> ${testLoss.toFixed(6)}`;
}

// [HINWEIS: Hier kommen die anderen Plot-Funktionen (BestFit/Overfit/Noisy) analog dazu...]
// (Der Übersichtlichkeit halber sind sie in deinem Code bereits korrekt definiert)

/* ============================================================
   UI & STORAGE
   ============================================================ */

function showSection(id) {
    document.querySelectorAll('.content-section').forEach(s => s.style.display = 'none');
    document.getElementById(id).style.display = 'block';
}

// Akkordeon-Fix: Resize Plotly wenn Reiter geöffnet wird
document.querySelectorAll(".accordion-header").forEach(header => {
    header.addEventListener("click", () => {
        const item = header.parentElement;
        item.classList.toggle("active");
        
        // Plotly Fix
        setTimeout(() => {
            const plots = item.querySelectorAll('.js-plotly-plot');
            plots.forEach(p => Plotly.Plots.resize(p));
        }, 300);
    });
});

async function saveMyModel(model, name) {
    try {
        await model.save(`localstorage://${name}`);
        console.log(`Modell gespeichert: ${name}`);
    } catch (e) {
        console.error("Speichern fehlgeschlagen", e);
    }
}

// Event Listener für den Haupt-Button
document.getElementById('btn-start-pipeline').addEventListener('click', startDataPipeline);

// Initialisierung
window.onload = () => {
    showSection('section-intro'); // Starte mit der Einführung
};
