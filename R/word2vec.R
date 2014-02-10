train.vectors <- function(train.file, output.file="vectors.bin", size=100,
                          cbow=0, hs=1, negative=0, min.count=5, alpha=0.025){
  .Call("train", train.file, output.file, as.integer(size),
        as.integer(cbow), as.integer(hs), as.integer(negative),
        as.integer(min.count), as.double(alpha)
        , PACKAGE="word2vec")
}

read.vectors <- function(vector.file) {
  .Call("load_vectors", vector.file)
}
