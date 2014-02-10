train.vectors <- function(trainFile, outputFile) {
  .Call("train", trainFile, outputFile)
}

read.vectors <- function(vectorFile) {
  .Call("load_vectors", vectorFile)
}
