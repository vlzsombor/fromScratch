module ConsoleApp1.DecisionTree

open System
open FSharp.Stats
//        best_split = {'gain':- 1, 'feature': None, 'threshold': None}

type Split() =
    let mutable leftSplit: Split option = None
    let mutable rightSplit: Split option = None
    let mutable gain : int = -1
    let mutable feature : int = -1
    let mutable threshold : int = -1
    member this.LeftSplit
        with get() = leftSplit
        and set(v) = leftSplit <- v
    member this.RightSplit
        with get() = rightSplit
        and set(v) = rightSplit <- v
    member x.Gain with get() = gain and set(v) = gain <- v
    member x.Feature with get() = feature and set(v) = feature <- v
    member x.Threshold with get() = threshold and set(v) = threshold <- v
    
   
type DecisionTree() =
    let minSample = 2
    let maxDebth = 2
    member this.split_data(dataset, feature, threshold) =
        
    member this.BestSplit(dataset:Matrix<double>, num_features:int) =
        let best_split = Split()
        for feature_index = 0 to num_features do
            let feature_values = Matrix.getCol dataset feature_index
            let unique = feature_values // ?? unique
            for threshold = 0 to unique.Length do
                
            
            
            
            printf "asafsdf"
        0.0
        
    member this.BuildTree(X:Matrix<double>, y: Vector<double>, current_depth: int) =
        let n_samples, n_features = X.Dimensions
        if n_samples >= minSample && currentDebth <= maxDebth then
            
    member this.Fit(X:Matrix<double>, y: Vector<double>) =
        let root = BuildTree(X, y)
        0
       
let dt = DecisionTree()



dt.Fit()