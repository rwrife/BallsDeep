
function Matrix(array, y) {
    this._matrix = [];

    if(typeof array == 'number' && y && typeof y == 'number') {
        this._matrix = [];
        for(var i=0;i<y;i++) {
            this._matrix.push(new Array(array));
        }
    } else {
        this._matrix = array;
    }

    this.shape = function() {
        return [this._matrix[0].length, this._matrix.length];
    }

    this.array = function() {
        return this._matrix;
    }

    this.add = function(b) {
        var a = this.map((x,y,e) => {            
            return e + ((typeof b == "number") ? b : b.val(x,y));
        });
        return a;
    }

    this.val = function(x,y) {
        return this._matrix[y][x];
    }

    this.neg = function() {
        var a = this.map((x,y,e) => {
            return 0-e;
        });
        return a;
    }

    this.map = function(f) {
        var mapArray = [this._matrix.length];
        for(var y=0;y<this._matrix.length;y++) {
             mapArray[y] = [this._matrix[y].length];
            for(var x=0;x<this._matrix[y].length;x++) {
                mapArray[y][x] = f(x,y, this._matrix[y][x]);
            }
        }
        return new Matrix(mapArray);        
    }

    this.fill = function(v) {
        var a = this.map((x,y,e) => {
            return v;
        });
        return a;
    }

    this.random = function() {
        var a = this.map((x,y,e) => {
            return Math.random();
        });
        return a;        
    }

    this.copy = function() {
        var t = new Matrix(this._matrix);
        return t;
    }

    this.dot = function(b)  {
        var m1 = this._matrix;

        if(this.shape()[0] != b.shape()[1]) //may need to transpose
            b = b.transpose();

        var m2 = b.array();

        var result = [];
        for (var i = 0; i < m1.length; i++) {
            result[i] = [];
            for (var j = 0; j < m2[0].length; j++) {
                var sum = 0;
                for (var k = 0; k < m1[0].length; k++) {
                    sum += m1[i][k] * m2[k][j];
                }
                result[i][j] = sum;
            }
        }
        return new Matrix(result);
    }

    this.exp = function() {
        var a = this.map((x,y,e) => {
            return Math.pow(Math.E, e);
        });
        return a;
    }

    this.inv = function() {
        //if(this._matrix.length != this._matrix[0].length) return null;
        var _A = this._matrix;
        var temp,
        N = _A.length,
        E = [];
   
        for (var i = 0; i < N; i++)
            E[i] = [];
   
        for (i = 0; i < N; i++)
            for (var j = 0; j < N; j++) {
                E[i][j] = 0;
                if (i == j)
                E[i][j] = 1;
            }
   
        for (var k = 0; k < N; k++) {
            temp = _A[k][k];

            for (var j = 0; j < N; j++) {
                _A[k][j] /= temp;
                E[k][j] /= temp;
            }

            for (var i = k + 1; i < N; i++) {
                temp = _A[i][k];
    
                for (var j = 0; j < N; j++) {
                    _A[i][j] -= _A[k][j] * temp;
                    E[i][j] -= E[k][j] * temp;
                }
            }
        }
   
        for (var k = N - 1; k > 0; k--) {
            for (var i = k - 1; i >= 0; i--) {
                temp = _A[i][k];
   
                for (var j = 0; j < N; j++) {
                    _A[i][j] -= _A[k][j] * temp;
                    E[i][j] -= E[k][j] * temp;
                }
            }
        }
   
        for (var i = 0; i < N; i++)
            for (var j = 0; j < N; j++)
                _A[i][j] = E[i][j];

        return new Matrix(_A);           
    }

    this.transpose = function() {
        var a = this._matrix; 
        var w = a.length || 0;
        var h = a[0] instanceof Array ? a[0].length : 0;

        if(h === 0 || w === 0) { return []; }

        var i, j, t = [];

        for(i=0; i<h; i++) {
            t[i] = [];
            for(j=0; j<w; j++) {
                t[i][j] = a[j][i];
            }
        }
        return new Matrix(t);
    }      
}

function sigmoid(m, isDeriv) {  
    if(!m instanceof Matrix) return;

    if(isDeriv) {
        return m.neg().add(1).map((x,y,e) => {
            return e * m.val(x,y);
        });
    } else {
        return m.neg().exp().add(1).map((x,y,e) => {
            return 1/e;
        });
    }
}

var syn0 = (new Matrix(3,1)).random().map((x,y,e) => {
    return e*2;
}).map((x,y,e) => {
    return e-1;
});

X = new Matrix([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ]);

y = (new Matrix([[0,0,1,1]])).transpose(); //labels

for (var i=0;i<10000;i++) {

    // forward propagation
    var l0 = X;
    var l1 = sigmoid(l0.dot(syn0), false);

    // how much did we miss?
    var l1_error = y.map((x,y,e) => {
        return e - l1.val(x,y);
    });

     // multiply how much we missed by the 
    // slope of the sigmoid at the values in l1
    l1_delta = sigmoid(l1, true).map((x,y,e) => {
        return e * l1_error.val(x,y);
    });

    // update weights
    syn0 = syn0.add(l0.transpose().dot(l1_delta).transpose());
}

console.log ("Output After Training:")
console.log (l1.array())

