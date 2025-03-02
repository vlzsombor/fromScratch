// For more information see https://aka.ms/fsharp-console-apps
open FSharp.Stats
open FSharp.Data
open System
//np.sum(np.square(predictions - self.y)) / (2 * m)

type LinearRegression() =
    static let compute_cost (y_pred: float[]) (y_true: float[]) : float =
        let m = float (Array.length y_pred)

        let cost =
            Array.fold2 (fun acc x y -> (acc + (x - y) ** 2.) / (2. * m)) 0. y_pred y_true

        cost

    static let backward (prediction: float[]) (y_true: float[]) (x_array: float[])=
        let m = float (Array.length prediction)
        let dB = Array.fold2 (fun acc x x2 -> (acc + x - x2) / m) 0. prediction y_true
        
        let temp = Array.map2 (-) prediction y_true
       
        let dW = Array.fold2 (fun acc x x2 -> (acc + x * x2) / m) 0. temp x_array
        dW, dB

    static let rec GradientDescent
        (X: float[])
        (y: float[])
        (w: float, b: float)
        (iteration: int)
        (prevCost)
        (alpha: float)
        =
        let prediction = LinearRegression.predict X w b
        let cost = compute_cost prediction y
        let dW, dB = backward prediction y X
        printfn "%O %O" (w,b) cost
        if iteration <= 0 || Math.Abs(prevCost - cost) < 0.0000000000001 then
            w - (dW * alpha), b - (dB * alpha)
        else
            GradientDescent X y (w - (dW * alpha), b - (dB * alpha)) (iteration - 1) cost alpha

    static member CalculateGradientDescent
        (X: float[])
        (y: float[])
        (w: float, b: float)
        (iteration: int)
        (alpha: float)
        =
        GradientDescent X y (w, b) iteration 10000000 alpha

    static member init = (Random.rndgen.NextFloat(), 0.)

    static member root_mean_squared_error (xInput: float[]) (x_realInput: float[]) =
        Math.Sqrt(LinearRegression.MSE xInput x_realInput)

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


let floatArray (row: seq<CsvRow>) (index: int) =
    row
    |> Seq.choose (fun row ->
        try
            Some(float (row.Item index)) // Try to convert
        with _ ->
            None // Skip bad rows
    )
    |> Seq.toArray



let train =
    CsvFile.Load("/Users/zsomborveres-lakos/git/ds/fromScratch/linearRegression/kaggle/train.csv")

let trainXtemp = (floatArray train.Rows 0)
let trainY = floatArray train.Rows 1


let test =
    CsvFile.Load("/Users/zsomborveres-lakos/git/ds/fromScratch/linearRegression/kaggle/test.csv")

let testXtemp = (floatArray test.Rows 0)
let testY = floatArray test.Rows 1
let trainX, testX = LinearRegression.standardize_data trainXtemp testXtemp

let w, b = LinearRegression.coefficient_ols trainX trainY

printfn "%O" (LinearRegression.MSE (LinearRegression.predict testX w b) testY)
printfn "%O" (LinearRegression.root_mean_squared_error (LinearRegression.predict testX w b) testY)
printfn "%O" (LinearRegression.r_squared (LinearRegression.predict testX w b) testY)
printfn "%A" (w, b)

let wG, bG = LinearRegression.CalculateGradientDescent trainX trainY (LinearRegression.init) 1000000000 0.001
 
printfn "%O" (wG, bG)
printfn "%O" (LinearRegression.MSE (LinearRegression.predict testX wG bG) testY)
printfn "%O" (LinearRegression.root_mean_squared_error (LinearRegression.predict testX wG bG) testY)
printfn "%O" (LinearRegression.r_squared (LinearRegression.predict testX wG bG) testY)


