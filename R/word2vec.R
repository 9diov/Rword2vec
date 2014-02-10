train.vectors <- function(train.file, output.file="vectors.bin", layer1.size=100,
                          cbow=0, hs=1, negative=0, min.count=5, alpha=0.025,
                          read.vocab.file=NULL, save.vocab.file=NULL) {
  .Call("train", train.file, output.file, as.integer(layer1.size),
        as.integer(cbow), as.integer(hs), as.integer(negative),
        as.integer(min.count), as.double(alpha)
        , PACKAGE="word2vec")
}

read.vectors <- function(vectorFile) {
  .Call("load_vectors", vectorFile)
}
