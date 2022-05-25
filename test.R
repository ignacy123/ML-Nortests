#!/usr/bin/env Rscript
library(nortest)

args = commandArgs(trailingOnly=TRUE)
input_file = args[1]

d = read.table(input_file, skip=2)
expected = d[,ncol(d)-1]
d = d[,-ncol(d)]
d = d[,-ncol(d)]

tests = list(list(lillie.test, "Lillie"), 
             list(ad.test, "Ad"), 
             list(pearson.test, "Pearson"), 
             list(cvm.test, "Cvm"), 
             list(sf.test, "Sf"))

for (test in tests) {
  guess = apply(d, 1, function(x) test[[1]](x)$p.value)
  count_corr = function(alpha) {
    guess[guess >= alpha] = 1
    guess[guess != 1] = 0
    nrow(d) - sum(bitwXor(guess, expected))
  }
  res = optimize(count_corr, c(0, 1), maximum=TRUE)
  cat(test[[2]], "| correct:", res$objective, "| optimal p-value:", res$maximum, "\n")
}