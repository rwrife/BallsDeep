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