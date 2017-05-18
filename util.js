exports.importCSV = function(file) {
    var parse = require('csv-parse/lib/sync');
    var fs = require('fs');
    var contents = fs.readFileSync(file).toString();
    var records = parse(contents, {columns: true});
    return records;
}

exports.peelProperty = function(obj, propertyName) {
    if(Array.isArray(obj)) {
        var newObjs = [];
        for(i in obj) {
            if(obj[i].hasOwnProperty(propertyName)) {
                var newObj = {};
                newObj[propertyName] = obj[i][propertyName];
                delete obj[i][propertyName];
                newObjs.push(newObj);
            }
        }
        return newObjs;
    } else {
        if(obj.hasOwnProperty(propertyName)) {
            var newObj = {};
            newObj[propertyName] = obj[propertyName];
            delete obj[propertyName];
            return newObj
        }
    }
};

exports.dropProperty = function(obj, propertyName) {
    if(Array.isArray(obj)) {
        for(i in obj) {
            if(obj[i].hasOwnProperty(propertyName)) {
                delete obj[i][propertyName];
            }
        }
    } else {
        if(obj.hasOwnProperty(propertyName)) {
            delete obj[propertyName];
        }
    }
};

exports.objectArrayToMatrixArray = function(objs) {
    var rows = [];
    var fields = [];
    if(Array.isArray(objs)) {
        for(i in objs) {
            for(key in objs[i]) {
                if(!fields.includes(key)) {
                    fields.push(key);
                }
            }
        }

        for(i in objs) {
            var newRow = [fields.length];
            for(key in objs[i]) {
                var n = parseFloat(objs[i][key].trim());
                newRow[fields.indexOf(key)] = n || null;
            }            
            rows.push(newRow);
        }
    }
    return rows;
}