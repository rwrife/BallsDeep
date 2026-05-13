var Matrix = require('./matrix.js');

var util = require('./util.js');
var el = util.importCSV('periodic.csv');
util.dropProperty(el, "Discoverer");
util.dropProperty(el, "Year of Discovery");
util.dropProperty(el, "Electron Configuration");
util.dropProperty(el, "Element");
util.dropProperty(el, "Most Stable Crystal");
util.dropProperty(el, "Type");
util.dropProperty(el, "Phase");
util.dropProperty(el, "Symbol");
util.dropProperty(el, "Display Row");
util.dropProperty(el, "Display Column");

var atomicNums = util.objectArrayToMatrixArray(util.peelProperty(el, "Atomic Number"));
var elems = util.objectArrayToMatrixArray(el);

//setup some random weights
var syn0 = (new Matrix(elems[0].length, 1)).random().map((x, y, e) => {
    return e * 2;
}).map((x, y, e) => {
    return e - 1;
});

//our feature data
X = (new Matrix(elems)).fillEmpty(0);

//our label data (ie the stuff we'll predict later)
y = (new Matrix(atomicNums)).transpose();

//do 10k iterations, testing different weights
for (var i = 0; i < 10000; i++) {

    // forward propagation
    var l0 = X;
    var l1 = sigmoid(l0.dot(syn0), false);

    // how much did we miss?    
    var l1_error = y.transpose().map((x, y, e) => {
        return e - l1.val(x, y);
    });

    // multiply how much we missed by the 
    // slope of the sigmoid at the values in l1    
    l1_delta = sigmoid(l1, true).map((x, y, e) => {
        // console.log(x, y, e)
        return e * l1_error.val(x, y);
    });

    // update weights
    syn0 = syn0.add(l0.transpose().dot(l1_delta).transpose());
}


/*var test = new Matrix([[1,1,0]]);
var l1 = sigmoid(test.dot(syn0), false);*/

console.log("Output After Training:")
console.log(l1.array())