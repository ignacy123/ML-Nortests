#!/usr/bin/env Rscript
library(functional)
library(nortest)
library(Pareto)
library(extraDistr)

args = commandArgs(trailingOnly=TRUE)
count = as.integer(args[1])
n = as.integer(args[2])
org_width = as.integer(args[3])
file_conn = file(args[4], open = "wt")
precision = 20
writeLines(c(as.character(count), as.character(n)), file_conn)

distributions = list(
  list(function(width) {
    left = runif(1, -width, width)[[1]]
    right = runif(1, left, width)[[1]]
    runif(n, left, right)
  }, "uniform"),
  list(function(width) rbeta(n, shape1 = runif(1, 0, width)[[1]], shape2 = runif(1, 0, width)[[1]]), "beta"),
  list(function(width) rbeta(n, shape1 = runif(1, 0, 1)[[1]], shape2 = runif(1, 0, 1)[[1]]), "beta2"),
  list(function(width) rcauchy(n, location = runif(1, -width, width)[[1]], scale = runif(1, 0, 1)[[1]]), "cauchy"),
  list(function(width) rcauchy(n, location = runif(1, -width, width)[[1]], scale = runif(1, 0, 100)[[1]]), "cauchy2"),
  list(function(width) rt(n, rdunif(1, 1, 10)[[1]]), "t_student"),
  list(function(width) rPareto(n, t = runif(1, 0, width)[[1]], alpha = runif(1, 0, 1)[[1]]), "pareto"),
  list(function(width) rPareto(n, t = runif(1, 0, width)[[1]], alpha = runif(1, 0, 50)[[1]]), "pareto2"),
  list(function(width) extraDistr::rgumbel(n, mu = runif(1, -width, width)[[1]], sigma = runif(1, 0, 10)[[1]]), "gumbel"),
  list(function(width) extraDistr::rgumbel(n, mu = runif(1, -width, width)[[1]], sigma = runif(1, 0, width)[[1]]), "gumbel2")
)

normals = list(
  list(function(width) rnorm(n, mean = runif(1, -width, width)[[1]], sd = runif(1, 0, width)[[1]]), "normal")
)

all_equal = function(x) {
  for (i in 1:length(x)) {
    if (round(x[i], precision) != round(x[1], precision)) return(FALSE)
  }
  TRUE
}

for (i in 1:(as.integer(count/2))) {
  width = runif(1, 0, 2*org_width)[[1]]
  distribution_pair = distributions[[rdunif(1, 1, length(distributions))[[1]]]]
  probe = distribution_pair[[1]](width)
  while (-Inf %in% probe || Inf %in% probe || all_equal(probe)) {
    probe = distribution_pair[[1]](width)
  }
  probe = append(round(probe, precision), 0)
  probe = append(round(probe, precision), distribution_pair[[2]])
  writeLines(c(paste(as.character(probe), collapse = " ")), sep = "\n", file_conn)
}

for (i in 1:(count - as.integer(count/2))) {
  width = runif(1, 0, 2*org_width)[[1]]
  distribution_pair = normals[[rdunif(1, 1, length(normals))[[1]]]]
  probe = distribution_pair[[1]](width)
  while (-Inf %in% probe || Inf %in% probe || all_equal(probe)) {
    probe = distribution_pair[[1]](width)
  }
  probe = append(round(probe, precision), 1)
  probe = append(round(probe, precision), distribution_pair[[2]])
  writeLines(c(paste(as.character(probe), collapse = " ")), sep = "\n", file_conn)
}
close(file_conn)
