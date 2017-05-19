var BD = require('./ballsdeep.js');
BD.loadData('./binary.csv', 'label');
BD.NonLinearNetwork.train();

console.log("Output After Training:")
console.log(BD.results.array())