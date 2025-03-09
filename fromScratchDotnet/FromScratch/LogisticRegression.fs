module ConsoleApp1.LogisticRegression

open FSharp.Stats
open System
open FSharp.Stats.Distributions.Empirical
open FSharp.Stats.Fitting.NonLinearRegression.Table.GrowthModels
open FSharp.Stats.Matrix
open FSharp.Stats.SpecializedGenericImpl
open FSharp.Stats.Vector
open FSharp.Stats.Vector.Generic
open Microsoft.FSharp.Core

type LogisticRegression() =
    static let compute_cost (y_pred: float[]) (y_true: float[]) : float =
        let m = float (Array.length y_pred)
        let cost = Array.fold2 (fun acc x y -> (acc + (x - y) ** 2.)) 0. y_pred y_true
        (cost / (2. * m))

    static let backward (prediction: float[]) (y_true: float[]) (x_array: float[]) =
        let m = float (Array.length prediction)
        let dB = (Array.fold2 (fun acc x x2 -> (acc + x - x2)) 0. prediction y_true) / m

        let temp = Array.map2 (-) prediction y_true

        let dW = (Array.fold2 (fun acc x x2 -> (acc + x * x2)) 0. temp x_array) / m
        dW, dB

    static let rec GradientDescent
        (X: float[])
        (y: float[])
        (w: float, b: float)
        (iteration: int)
        (prevCost)
        (alpha: float)
        =
        let prediction = LogisticRegression.predict X w b
        let cost = compute_cost prediction y
        let dW, dB = backward prediction y X
        //printfn "%O %O" (w,b) cost
        if iteration <= 0 then
            w - (dW * alpha), b - (dB * alpha)
        else
            GradientDescent X y (w - (dW * alpha), b - (dB * alpha)) (iteration - 1) cost alpha

    static let rec GradientDescent2
        (X: float[])
        (y: float[])
        (w: float, b: float)
        (iteration: int)
        (prevCost)
        (alpha: float)
        =

        let prediction = LogisticRegression.predict X w b
        let cost = compute_cost prediction y
        let dW, dB = backward prediction y X
        //printfn "%O %O" (w,b) cost
        seq {
            if iteration <= 0 || (Math.Abs(cost - prevCost)) < 0.000000000001 then
                yield w, b
            else
                yield w, b
                yield! GradientDescent2 X y (w - (dW * alpha), b - (dB * alpha)) (iteration - 1) cost alpha
        }

    static member CalculateGradientDescent
        (X: float[])
        (y: float[])
        (w: float, b: float)
        (iteration: int)
        (alpha: float)
        =
        GradientDescent2 X y (w, b) iteration 10000000 alpha

    static member init = (Random.rndgen.NextFloat(), 0.)

    static member root_mean_squared_error (xInput: float[]) (x_realInput: float[]) =
        Math.Sqrt(LogisticRegression.MSE xInput x_realInput)

    static member coefficient_ols (X: float[]) (Y: float[]) = // Ordinary least squares
        let n = X.Length
        let x_mean = Array.average X
        let y_mean = Array.average Y

        let numerator =
            Array.fold2 (fun acc x y -> acc + (x - x_mean) * (y - y_mean)) 0. X Y

        let denominator = Array.fold (fun acc x -> acc + (x - x_mean) ** 2.0) 0. X

        let slope = numerator / denominator
        let intercept = y_mean - slope * x_mean
        slope, intercept

    static member predict (x: float[]) (w: float) (b: float) : float[] = x |> Array.map (fun xi -> w * xi + b)

    static member standardize_data (x_train: float[]) (x_test: float[]) =
        let mean = x_train |> Seq.meanBy float
        let std = Seq.stDevPopulation x_train
        let predict (x: float[]) (w: float) (b: float) : float[] = x |> Array.map (fun xi -> w * xi + b)

        let calculation (x: float[]) =
            x |> Array.map (fun xi -> (xi - mean) / std)

        let retXTrain = calculation x_train
        let retYTrain = calculation x_test
        retXTrain, retYTrain

    static member MSE (xInput: float[]) (x_realInput: float[]) =
        (Array.fold2 (fun acc x x_r -> acc + (x - x_r) ** 2.0) 0. xInput x_realInput)
        / float xInput.Length

    static member r_squared (y_pred: float[]) (y_true: float[]) =
        let mean_y = y_true |> Seq.meanBy float

        let ss_total = Array.fold (fun acc x -> acc + ((x - mean_y)) ** 2.0) 0. y_true

        let ss_residual =
            Array.fold2 (fun acc x xpred -> acc + ((x - xpred)) ** 2.) 0. y_true y_pred

        let r2 = 1. - (ss_residual / ss_total)
        r2


type IGradientDescent =
    abstract member CostFunction: Vector<float> -> float
    abstract member BackwardPass: Vector<float> -> Vector<float>*float
    abstract member ForwardPass: Matrix<float> -> Vector<float>

type MyDisposable() =
    member val W: Vector<float> = MatrixTopLevelOperators.vector [ 0.0 ] with get, set
    member val b: float = 1. with get, set
    member val Y: Vector<float> = MatrixTopLevelOperators.vector [ 0.0 ] with get, set
    member val X: Matrix<float> = MatrixTopLevelOperators.matrix [ [0.0] ] with get, set

    static member sigmoid z : Vector<float> =
        map (fun x -> 1. / (1. + (Math.Exp -x))) z
// havent tested
    interface IGradientDescent with
        member this.CostFunction(predictions) : float =
            let smalloffset = 0.00000001
            let cost =
                sum (
                    map2
                        (fun x y -> y * Math.Log(x + smalloffset) + (1. - y) * Math.Log(1. - x + smalloffset))
                        predictions
                        this.Y
                )
            - cost / float predictions.Length
        member this.BackwardPass(predictions: Vector<float>)=
            let dW = (mulMV this.X.Transpose (sub predictions this.Y))
            let dB = mean (sub predictions this.Y)
           
            let dW2 = scale (1./float predictions.Length) dW 
            dW2, dB

        member x.ForwardPass(X: Matrix<float>) : Vector<float> =
            MyDisposable.sigmoid (addScalarV x.b (mulV X x.W))

    member this.Fit(X,y, iterations) =
        let learningRate = 0.0001
        
        this.X <- X
        this.Y <- y
        this.InitializeParams X
        
        for i = 0 to iterations do
            let preds = (this :> IGradientDescent).ForwardPass(X)
            let cost = (this:> IGradientDescent).CostFunction(preds)
            let dW,dB = (this:> IGradientDescent).BackwardPass(preds)
            this.W <- this.W - learningRate * dW
            this.b <- this.b - learningRate * dB
            
            if i % 10000 = 0 then
                printfn $"Cost after iteration %A{cost}"

    member this.InitializeParams(X: Matrix<float>) =
        this.b <- 0
        this.W <- MatrixTopLevelOperators.vector (Array.init X.NumCols (fun x -> 0))


let y = MatrixTopLevelOperators.vector [7; 1; -2; 3]
let X: Matrix<double> = matrix [ [1;2;3]; [4;5;6]; [7;8;9];[10;11;17] ]

let aaa = MyDisposable()
// let aadfasd = a.NumRows
// let afasdfasfasd = MatrixTopLevelOperators.vector (Array.init 3 (fun x -> 0))
// aaa.InitializeParams(a)
//
//


aaa.Fit(X,y,1000)

printfn "%A" aaa.W
//printfn "%A" ((aaa :> (IGradientDescent)).ForwardPass (a))


// type DictionaryBackedClass() =
//     inherit Node("fads")
//     let dict = System.Collections.Generic.Dictionary<string, string>()
//
//     override this.GetArea() = 3.0
//
