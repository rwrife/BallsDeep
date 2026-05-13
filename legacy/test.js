var BD = require('./ballsdeep.js');
BD.loadData('./binary.csv', 'label');
BD.NonLinearNetwork.train();

console.log("Output After Training:")
console.log(BD.results.array())

/*
var act = require('./activation.js');
var matrix = require('./matrix.js');
var m = new matrix([
    [0.1,   -0.1],
    [0.2,   0.2]
]);

console.log(act.softmax(m).array())
*/