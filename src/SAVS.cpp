#include <RcppArmadillo.h>
#include "SAVS.h"
#include <progress.hpp>

using namespace Rcpp;
using namespace arma;

arma::rowvec savs(const arma::mat X, const arma::rowvec b, const double nu, 
                  const arma::rowvec pen){
  
  arma::rowvec mu = pen / abs(pow(b, nu)); 
  arma::rowvec normalizer = sum(square(X), 0);
  arma::rowvec b_sparse = sign(b) / normalizer % ((abs(b % normalizer) - mu));
  b_sparse.elem(find((abs(b)%normalizer)<mu)).zeros();
  
  return(b_sparse);
}

RcppExport SEXP DSAVS(const SEXP fsvdraws_in){
  
  const List fsvdraws(fsvdraws_in);
  // extract factors, factorloadings and log variances
  arma::cube Fac_draws = fsvdraws["fac"];
  arma::cube Facload_draws = fsvdraws["facload"];
  arma::cube logvar_draws = fsvdraws["logvar"];
  const int m = Facload_draws.n_rows;
  const int r = Facload_draws.n_cols;
  arma::cube Faclogvar_draws = logvar_draws.cols(m, m + r-1);
  
  // initialization of some objects
  const int draws = Facload_draws.n_slices;
  const int n = Fac_draws.n_cols;
  arma::mat Fac(n,m);
  arma::mat Fac_norm(size(Fac));
  arma::mat Fac_norm_perm_t(size(Fac));
  arma::mat normalizer(size(Fac));
  arma::mat Facload(m,r);
  arma::mat Facload_t(size(Facload));
  arma::mat Facload_perm_t(size(Facload));
  arma::mat Facload_sparse_perm_t(size(Facload));
  arma::uvec perm_t(r);
  arma::vec eigs_t(r);
  arma::vec penalty_t(r);
  
  // storage
  arma::mat active_sparse_store(n,draws);
  arma::cube eigs_store(n, r, draws);
  
  // Initialize progressbar
  Progress p(draws, true);
  for(int rep = 0; rep < draws; rep++){
    
    Fac = trans(Fac_draws.slice(rep));
    normalizer = exp(Faclogvar_draws.slice(rep)/2);
    Fac_norm = Fac / normalizer;
    Facload = Facload_draws.slice(rep);
    
    for(int t = 0; t < n; t++){
      
      Facload_t = Facload.each_row() % normalizer.row(t);
//      for(int i = 0; i < m; i++){
////        
//        for(int j = 0; j < r; j++){
//          
//          Facload_t(i, j) = Facload(i, j) * normalizer(t, j);
//          
//        }
//        
//        
//        
//        
//      }
      
      arma::rowvec Facload_t_sums = sum(abs(Facload_t), 0);
      perm_t = sort_index(Facload_t_sums, "ascend");
      Facload_perm_t = Facload_t.cols(perm_t);
      Fac_norm_perm_t = Fac_norm.cols(perm_t);
      
      arma::mat Facload_t_crossprod = Facload_t.t() * Facload_t;
      arma::mat eigvec_t;
      eig_sym(eigs_t, eigvec_t, Facload_t_crossprod);
      eigs_store( span(t, t), span(0, r-1), span(rep, rep)) = eigs_t;
      penalty_t = max(eigs_t) / eigs_t;
      
      for(int i = 0; i < m; i++){
        
        Facload_sparse_perm_t.row(i) = savs(Fac_norm_perm_t, 
                                  Facload_perm_t.row(i), 2, penalty_t.t());
        
      }
      
      arma::rowvec tmp = sum(abs(Facload_sparse_perm_t), 0);
      int n_active = 0;
      for(int i = 0; i < r; i++){
        if(tmp(i)>0){
          n_active++;
        }
      }
      active_sparse_store(t, rep) = n_active;
    }
    p.increment();
  }
  List out = List::create(
    Named("eigenvalues") = eigs_store,
    Named("active") = active_sparse_store
  );
  
  return out;
}

