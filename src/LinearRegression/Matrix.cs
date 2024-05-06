// Project: DIY Neural Network
// File name: Matrix.cs
// Created by: dotnet.org.pl 2024

using System.Diagnostics;

internal class Matrix
{
    private readonly Array _array;

    public Matrix(Array array)
    {
        _array = array;
    }

    public Matrix(int rows, int columns)
    {
        _array = Array.CreateInstance(typeof(float), rows, columns);
    }

    public Array Array => _array;

    // Create an implicite conversion to array.
    public static implicit operator Array(Matrix matrix) => matrix.Array;

    #region Zeros, Ones, and Random

    internal static Matrix Zeros(int rows, int columns) => new(rows, columns);

    internal static Matrix Ones(Matrix matrix)
    {
        (int rows, int columns) = GetDimensions(matrix);
        return Ones(rows, columns);
    }

    internal static Matrix Ones(int rows, int columns)
    {
        // Create an instance of Array of floats using rows and columns and fill it with ones.
        Array array = Array.CreateInstance(typeof(float), rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(1f, i, j);
            }
        }
        return new Matrix(array);
    }

    internal static Matrix Random(int rows, int columns, Random random)
    {
        // create an instance of Array of floats using rows and columns and fill it with randoms
        Array array = Array.CreateInstance(typeof(float), rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(random.NextSingle() - 0.5f, i, j);
            }
        }
        return new Matrix(array);
    }

    #endregion

    #region Operations with scalar

    internal Matrix Add(float scalar)
    {
        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                // ((float[,])array)[i, j] = ((float[,])_array)[i, j] + scalar;
                array.SetValue((float)_array.GetValue(i, j)! + scalar, i, j);
            }
        }

        return new Matrix(array);
    }

    internal Matrix Multiply(float scalar)
    {
        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue((float)_array.GetValue(i, j)! * scalar, i, j);
            }
        }

        return new Matrix(array);
    }

    internal Matrix Power(int scalar)
    {
        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(MathF.Pow((float)_array.GetValue(i, j)!, scalar), i, j);
            }
        }

        return new Matrix(array);
    }

    #endregion

    #region Operations with matrix

    internal Matrix MultiplyDot(Matrix matrix)
    {
        Debug.Assert(GetDimension(Dimension.Columns) == matrix.GetDimension(Dimension.Rows));

        // Get the number of rows of the second matrix.
        int matrixColumns = matrix.Array.GetLength(1);

        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        Array array = Array.CreateInstance(typeof(float), rows, matrixColumns);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < matrixColumns; j++)
            {
                float sum = 0;
                for (int k = 0; k < columns; k++)
                {
                    sum += (float)_array.GetValue(i, k)! * (float)matrix.Array.GetValue(k, j)!;
                }
                array.SetValue(sum, i, j);
            }
        }

        return new Matrix(array);
    }

    internal Matrix MultiplyElementwise(Matrix matrix)
    {
        Debug.Assert(GetDimension(Dimension.Rows) == matrix.GetDimension(Dimension.Rows));

        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue((float)_array.GetValue(i, j)! * (float)matrix.Array.GetValue(i, j)!, i, j);
            }
        }

        return new Matrix(array);
    }

    internal Matrix Subtract(Matrix matrix)
    {
        Debug.Assert(GetDimension(Dimension.Rows) == matrix.GetDimension(Dimension.Rows));
        Debug.Assert(GetDimension(Dimension.Columns) == matrix.GetDimension(Dimension.Columns));

        (Array array, int rows, int columns) = GetCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue((float)_array.GetValue(i, j)! - (float)matrix.Array.GetValue(i, j)!, i, j);
            }
        }

        return new Matrix(array);
    }

    #endregion

    #region Aggregations

    internal float Mean() => Sum() / _array.Length;

    internal float Sum()
    {
        // return _array.Cast<float>().Sum();
        // Sum over all elements.
        float sum = 0;
        foreach (object? item in _array)
        {
            sum += (float)item!;
        }
        return sum;
    }

    #endregion

    #region Slices and Rows

    internal Matrix GetRow(int row)
    {
        int columns = _array.GetLength(1);

        // Create an array to store the row.
        float[,] newArray = new float[1, columns]; 
        for (int i = 0; i < columns; i++)
        {
            // Access each element in the second row.
            newArray[0, i] = (float)_array.GetValue(row, i)!; 
        }

        return new Matrix(newArray);
    }

    internal void SetRow(int row, Matrix values)
    {
        Debug.Assert(values.GetDimension(Dimension.Columns) == _array.GetLength(1));

        for (int i = 0; i < _array.GetLength(1); i++)
        {
            _array.SetValue(values.Array.GetValue(0, i), row, i);
        }
    }

    internal Matrix GetRows(Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(_array.GetLength(0));

        Array newArray = Array.CreateInstance(typeof(float), length, _array.GetLength(1));

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                newArray.SetValue(_array.GetValue(i + offset, j), i, j);
            }
        }

        return new Matrix(newArray);
    }

    #endregion

    #region Matrix operations

    internal Matrix Transpose()
    {
        // create a new MyArray instance with the columns and rows swapped

        // get the number of rows of _array
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        Array array = Array.CreateInstance(typeof(float), columns, rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array.SetValue(_array.GetValue(i, j), j, i);
            }
        }

        return new Matrix(array);
    }

    #endregion

    internal int GetDimension(Dimension dimension) => _array.GetLength((int)dimension);

    private static (int Rows, int Columns) GetDimensions(Matrix inputMatrix) => (inputMatrix.GetDimension(Dimension.Rows), inputMatrix.GetDimension(Dimension.Columns));

    private (Array Array, int Rows, int Columns) GetCopyAsArray()
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);
        Array array = Array.CreateInstance(typeof(float), rows, columns);
        return (array, rows, columns);
    }

    // Some commented out code experimenting with different ways to implement the indexer, and slicing.

    //internal MyArray GetSlice(int dimension, int index)
    //{
    //    // if the dimension is 0 then return the column
    //    if (dimension == 0)
    //    {
    //        return new MyArray(_array.GetValue(0, index) as Array);
    //    }

    //    // if the dimension is 1 then return the row
    //    return new MyArray(_array.GetValue(index, 0) as Array);
    //}

    //internal MyArray this[params int[] index]
    //{
    //    get
    //    {
    //        float[] newArray = new float[_array.GetLength(1)]; // Create an array to store the second row
    //        for (int i = 0; i < _array.GetLength(1); i++)
    //        {
    //            newArray[i] = (float)_array.GetValue(index[0], i); // Access each element in the second row
    //        }

    //        // return a new MyArray instance with the specified index
    //        return new MyArray(newArray);
    //    }
    //    set
    //    {

    //        // set the value of the specified index
    //        _array.SetValue(value.Array, index);
    //    }
    //}

    //internal MyArray this[Range range]
    //{
    //    get
    //    {
    //        (int offset, int length) = range.GetOffsetAndLength(_array.GetLength(0));

    //        Array newArray = Array.CreateInstance(typeof(float), length, _array.GetLength(1));

    //        Array.Copy(_array, offset, newArray, 0, length);

    //        return new MyArray(newArray);
    //    }
    //    //set
    //    //{
    //    //    throw new NotImplementedException();
    //    //}
    //}
}