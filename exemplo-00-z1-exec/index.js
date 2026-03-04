import tf from '@tensorflow/tfjs';

const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels

const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba" }

const pessoaTensorNormalizado = [
    [
        0.34, // idade normalizada
        0,    // cor azul
        0,    // cor vermelho
        0,    // cor verde
        0,    // localização São Paulo
        0,    // localização Rio
        0     // localização Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results)

async function predict(model, pessoa) {
    const inputTensor = tf.tensor2d(pessoa)
    const prediction = model.predict(inputTensor)
    const predArray = await prediction.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}


async function trainModel(xs, ys) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [7], units: 10000, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(xs, ys, {
        epochs: 100,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    return model;
}
