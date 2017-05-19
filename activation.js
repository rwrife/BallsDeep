var Matrix = require('./matrix.js');

exports.sigmoid = function(m, isDeriv) {
    if (!m instanceof Matrix) return;

    if (isDeriv) {
        return m.neg().add(1).map((x, y, e) => {
            return e * m.val(x, y);
        });
    } else {
        return m.neg().exp().add(1).map((x, y, e) => {
            return 1 / e;
        });
    }
}

exports.bin = function(m) {
    if(!m instanceof Matrix) return;

    return m.map((x,y,e) => {
        return e < 0 ? 0 : 1;
    });    
}

exports.tanh = function(m) {
    if(!m instanceof Matrix) return;

    return m.map((x,y,e) => {
        return 2 / (1+Math.pow(Math.E, (0-2) * x)) - 1;
    });
}

exports.relu = function(m) {
    if(!m instanceof Matrix) return;

    return m.map((x,y,e) => {
        return e > 0 ? e : 0;
    });
}

exports.elu = function(m) {
    if(!m instanceof Matrix) return;

    return m.map((x,y,e) => {

    });
}

exports.softmax = function(m) {
    if(!m instanceof Matrix) return;
    
    var eSum = []    

    return (m.map((x,y,e) => {
        return Math.exp(e);
    })).transpose().map((x,y,e) => {
        if(typeof eSum[y] === 'undefined') eSum[y] = 0; //WTF??
        eSum[y] += e;        
        return e;
    }).map((x,y,e) => {
        return e / eSum[y];
    }).transpose();
}


