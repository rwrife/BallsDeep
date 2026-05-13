var Matrix = require('./matrix.js');
var act = require('./activation.js');
var util = require('./util.js');
var bin = util.importCSV('./binary.csv');
var label = util.objectArrayToMatrixArray(util.peelProperty(bin, 'label'));
var bins = util.objectArrayToMatrixArray(bin);

var syn0 = (new Matrix(bins[0].length, 1)).random().map((x, y, e) => {
    return e * 2;
}).map((x, y, e) => {
    return e - 1;
});

X = (new Matrix(bins)).fillEmpty(0);
y = (new Matrix(label)).transpose();

for (var i = 0; i < 10000; i++) {
    var l0 = X;
    var l1 = act.sigmoid(l0.dot(syn0), false);

    var l1_error = y.transpose().map((x, y, e) => {
        return e - l1.val(x, y);
    });

    l1_delta = act.sigmoid(l1, true).map((x, y, e) => {
        return e * l1_error.val(x, y);
    });

    syn0 = syn0.add(l0.transpose().dot(l1_delta).transpose());
}
console.log(BD.fields);
console.log("Output After Training:")
console.log(l1.array())