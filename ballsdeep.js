"use strict";

var Matrix = require('./matrix.js');
var util = require('./util.js');

function LinearNetwork() {
    this.labels = null;
    this.fields = null;

    this.loadData = function(csvFile, labelField) {
        var util = require('./util.js');
        var objData = util.importCSV(csvFile);

        this.setData(objData, labelField);
    };

    this.setData = function(objCollection, labelField) {
        this.labels = util.objectArrayToMatrixArray(util.peelProperty(objCollection, labelField));
        this.fields = util.objectArrayToMatrixArray(objCollection);

        this.X = (new Matrix(this.fields)).fillEmpty(0); //needs a mean fill
        this.y = (new Matrix(this.labels)).transpose(); //correct labels for calulation
    };

    this.train = function(iterations, onDone) {
        var act = require('./activation.js');

        var syn0 = (new Matrix(this.fields[0].length, 1)).random().map((x, y, e) => {
            return e * 2;
        }).map((x, y, e) => {
            return e - 1;
        });
        const itnum = iterations || 10000;
        for (var i = 0; i < itnum; i++) {
            var l0 = this.X;
            var l1 = act.sigmoid(l0.dot(syn0), false);

            var l1_error = this.y.transpose().map((x, y, e) => {
                return e - l1.val(x, y);
            });

            var l1_delta = act.sigmoid(l1, true).map((x, y, e) => {
                return e * l1_error.val(x, y);
            });

            syn0 = syn0.add(l0.transpose().dot(l1_delta).transpose());
        }

        this.results = l1;

        if (onDone) onDone(l1);
    }

    this.results = null;
}

module.exports = new LinearNetwork();