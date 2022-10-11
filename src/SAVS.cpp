#include <RcppArmadillo.h>
#include "SAVS.h"
#include <progress.hpp>

using namespace Rcpp;
using namespace arma;

arma::uvec match(arma::uvec x, arma::uvec y){
  int n = x.n_elem;
  arma::uvec ind(size(x));
  for(int i = 0; i<n; i++){
    
    uvec tmp = find(y == x(i), 1);
    ind(i) = tmp(0);
  }
  return ind;
}

arma::rowvec savs(const arma::mat X, const arma::rowvec b, const double nu, 
                  const arma::rowvec pen){
  
  arma::rowvec mu = pen / abs(pow(b, nu)); 
  arma::rowvec normalizer = sum(square(X), 0);
  arma::rowvec b_sparse = sign(b) / normalizer % ((abs(b % normalizer) - mu));
  b_sparse.elem(find((abs(b)%normalizer)<mu)).zeros();
  
  return(b_sparse);
}

// This function computes a sparse loading matrix for time t given one sample 
// of the joint posterior of all relevant parameters. 
void facload_sparse(arma::mat& Facload_sparse_t,
                    arma::vec& eigs_t,
                    int& n_active,
                    const arma::mat Fac,
                    const arma::mat Faclogvar,
                    const arma::mat Facload,
                    const int t,
                    const arma::uvec match_ind){
  
  const int m = Facload_sparse_t.n_rows;
  
  arma::mat normalizer = exp(Faclogvar/2);
  arma::mat Fac_norm = Fac / normalizer;
  
  arma::mat Facload_t = Facload.each_row() % normalizer.row(t);
  
  arma::rowvec Facload_t_sums = sum(abs(Facload_t), 0);
  arma::uvec perm = sort_index(Facload_t_sums, "ascend");
  arma::uvec reorder = match(match_ind, perm);
  arma::mat Facload_perm_t = Facload_t.cols(perm);
  arma::mat Fac_norm_perm = Fac_norm.cols(perm);
  
  arma::mat Facload_t_crossprod = Facload_t.t() * Facload_t;
  
  arma::mat eigvec_t;
  eig_sym(eigs_t, eigvec_t, Facload_t_crossprod);
  arma::vec penalty_t = max(eigs_t) / eigs_t;
  
  arma::mat Facload_sparse_perm_t(size(Facload));
  for(int i = 0; i < m; i++){
    
    Facload_sparse_perm_t.row(i) = savs(Fac_norm_perm, 
                              Facload_perm_t.row(i), 2, penalty_t.t());
  }
  Facload_sparse_t = Facload_sparse_perm_t.cols(reorder);
  
  arma::rowvec n_active_tmp = sum(abs(Facload_sparse_perm_t), 0);
  n_active_tmp.elem(find(n_active_tmp>0)).ones();
  n_active = accu(n_active_tmp);
  
}

RcppExport SEXP predsavs_cpp(const SEXP fsvdraws_in, const SEXP ahead_in){
  
  const List fsvdraws(fsvdraws_in);
  IntegerVector ahead_tmp(ahead_in);
  arma::ivec ahead(ahead_tmp.begin(), ahead_tmp.size(), false);
  const int ahead_max = max(ahead);
  
  // extract factors, factorloadings and log variances
  arma::cube facs = fsvdraws["fac"];
  arma::cube facloads = fsvdraws["facload"];
  arma::cube logvars = fsvdraws["logvar"];
  arma::mat y = fsvdraws["y"];
  
  // m: #time series, r: #factors, draws: #posterior draws, n: #observations
  const int m = facloads.n_rows;
  const int r = facloads.n_cols;
  const int draws = facloads.n_slices;
  const int n = y.n_rows;
  
  // extract sv parameters
  arma::cube sv_paras = fsvdraws["para"];
  arma::mat sv_mus = sv_paras.row(0);
  arma::mat sv_phis = sv_paras.row(1);
  arma::mat sv_sigmas = sv_paras.row(2);
  
//  int bottom = logvars.n_rows;
//  if(n!=bottom){
//    Rcpp::stop("This function needs the full history of logvariances. Run 'fsvsample' again and set 'keeptime='all''! ");
//  }
  
  // extract last logvar (needed to predict ahead)
  arma::mat logvar_preds = logvars.row(n-1);  
  
  //storage
  arma::cube Facload_sparse_pred_store(m*r, draws, ahead.size()); //ahead.size()
  arma::cube logvar_pred_store(m+r, draws, ahead.size());
  
  //init
  int n_active=0;
  arma::vec eigs_t(r);
  arma::mat Facload_sparse_pred(m,r);
  
  arma::cube logvars_plus_preds(n+ahead_max, m+r, draws); //ahead_max
  logvars_plus_preds.rows(0,(n-1)) = logvars;
  arma::cube facs_plus_preds(r, n+ahead_max, draws); //ahead_max
  facs_plus_preds.cols(0,(n-1)) = facs;

  int count = 0;
  for(int h = 0; h < ahead_max; h++){
    
    // generate some standard normal vectors
    arma::mat rnorm_mat_h(m+r,draws);
    for(int i = 0; i < (m+r); i++){
      for(int j = 0; j < draws; j++){
        rnorm_mat_h(i,j) = R::rnorm(0, 1);
      }
    }
    arma::mat rnorm_mat_f(r,draws);
    for(int i = 0; i < r; i++){
      for(int j = 0; j < draws; j++){
        rnorm_mat_f(i,j) = R::rnorm(0, 1);
      }
    }
    
    // h step ahead logvars and factors
    logvar_preds = sv_mus + sv_phis % 
      (logvar_preds - sv_mus) + sv_sigmas % rnorm_mat_h;
    if(h == ahead(count)-1){
      logvar_pred_store.slice(count) = logvar_preds;
    }
    arma::mat fac_preds = exp(logvar_preds.rows(m, m+r-1)/2) % rnorm_mat_f;
    
    // stack logvars and their predictions
    logvars_plus_preds.row(n+h) = logvar_preds; //+h
    
    // stack factors and their predictions
    facs_plus_preds.col(n+h) = fac_preds; //+h
    
    arma::uvec match_ind(r);
    for(int i = 0; i<r; i++){
      match_ind(i) = i;
    }
    
    for(int rep=0; rep<draws; rep++){
      arma::mat tmp0 = facs_plus_preds(span(), span(0,n+h), span(rep));
      arma::mat fac_pred = trans(tmp0);
      facload_sparse(Facload_sparse_pred,
                     eigs_t,
                     n_active,
                     fac_pred,
                     logvars_plus_preds( span(0, n+h), span(m, m+r-1), span(rep) ),
                     facloads.slice(rep),
                     n+h,//+h
                     match_ind);
      if(h == ahead(count)-1){
        Facload_sparse_pred_store(span(), span(rep), span(count)) = Facload_sparse_pred.as_col();
        count++; 
      }
          }
    
  }
  
  List out = List::create(
    Named("Facload_sparse_pred") = Facload_sparse_pred_store,
    Named("logvar_pred") = logvar_pred_store
  );
  
  return out;
  
}


RcppExport SEXP DSAVS2(const SEXP fsvdraws_in, const SEXP t_to_store){
  
  const List fsvdraws(fsvdraws_in);
  IntegerVector t_ind_tmp(t_to_store);
  arma::ivec t_ind(t_ind_tmp.begin(), t_ind_tmp.size() , false);
  const int n_t = t_ind.n_elem;
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
  arma::mat Facload(m,r);
  arma::mat Facload_sparse_t(size(Facload));
  arma::mat Faclogvar(size(Fac));
  arma::vec eigs_t(r);
  int n_active=0; //int
  
  arma::mat Facload_t_summary_tmp(m*r,draws);
  arma::colvec Facload_t_summary_tmp2(m*r);
  
  arma::uvec match_ind(r);
  for(int i = 0; i<r; i++){
    match_ind(i) = i;
  }
  
  // storage
  arma::mat active_sparse_store(n_t,draws);
  arma::cube eigs_store(n_t, r, draws);
  arma::cube Facload_sparse_t_store_tmp(m, r, draws);
  arma::cube Facload_sparse_t_summary(m, r, n_t);
  
  // Initialize progressbar
  Progress p(n_t*draws, true);
  arma::ivec::iterator t;
  int jj=0;
  for(t = t_ind.begin(); t != t_ind.end(); t++){
    for(int rep = 0; rep < draws; rep++){
      Fac = trans(Fac_draws.slice(rep));
      Faclogvar = Faclogvar_draws.slice(rep);
      Facload = Facload_draws.slice(rep);
      
      facload_sparse(Facload_sparse_t,
                     eigs_t,
                     n_active,
                     Fac,
                     Faclogvar,
                     Facload,
                     *t,
                     match_ind);
      
      active_sparse_store(jj, rep) = n_active;
      Facload_sparse_t_store_tmp.slice(rep) = Facload_sparse_t;
      eigs_store( span(jj, jj), span(0, r-1), span(rep, rep) ) = eigs_t;
      p.increment();
    }
    Facload_t_summary_tmp = reshape( mat(Facload_sparse_t_store_tmp.memptr(), Facload_sparse_t_store_tmp.n_elem, 1, false), m*r, draws);
    Facload_t_summary_tmp2 = arma::median(Facload_t_summary_tmp, 1); 
    Facload_sparse_t_summary.slice(jj) = reshape( mat(Facload_t_summary_tmp2.memptr(), Facload_t_summary_tmp2.n_elem, 1, true), m, r);
    
    jj++;
    p.increment();
  }
  List out = List::create(
    Named("eigenvalues") = eigs_store,
    Named("active") = active_sparse_store,
    Named("Facload_t_medians") = Facload_sparse_t_summary
  );
  
  return out;
}


RcppExport SEXP DSAVS(const SEXP fsvdraws_in, const SEXP t_to_store){
  
  const List fsvdraws(fsvdraws_in);
  IntegerVector t_ind_tmp(t_to_store);
  arma::ivec t_ind(t_ind_tmp.begin(), t_ind_tmp.size() , false);
  const int n_t = t_ind.n_elem;
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
  arma::mat Facload_t_crossprod(r,r);
  arma::rowvec Facload_t_sums(r);
  arma::rowvec n_active_tmp1(r);
  int n_active = 0;
  arma::uvec perm_t(r);
  arma::uvec reorder_t(r);
  arma::vec eigs_t(r);
  arma::mat eigvec_t(m,r);
  arma::vec penalty_t(r);
  
  arma::uvec match_ind(r);
  for(int i = 0; i<r; i++){
    match_ind(i) = i;
  }
  
  // storage
  arma::mat active_sparse_store(n,draws);
  arma::cube eigs_store(n, r, draws);
  arma::cube Facload_sparse_t_store_tmp(m, r, n_t);
  arma::field<cube> Facload_sparse_t_store(draws);
  
  // Initialize progressbar
  Progress p(draws, true);
  for(int rep = 0; rep < draws; rep++){
    
    Fac = trans(Fac_draws.slice(rep));
    normalizer = exp(Faclogvar_draws.slice(rep)/2);
    Fac_norm = Fac / normalizer;
    Facload = Facload_draws.slice(rep);
    
    int j=0;
    for(int t = 0; t < n; t++){
      
      Facload_t = Facload.each_row() % normalizer.row(t);
      
      Facload_t_sums = sum(abs(Facload_t), 0);
      perm_t = sort_index(Facload_t_sums, "ascend");
      reorder_t = match(match_ind, perm_t);
      Facload_perm_t = Facload_t.cols(perm_t);
      Fac_norm_perm_t = Fac_norm.cols(perm_t);
      
      Facload_t_crossprod = Facload_t.t() * Facload_t;
      
      eig_sym(eigs_t, eigvec_t, Facload_t_crossprod);
      eigs_store( span(t, t), span(0, r-1), span(rep, rep)) = eigs_t;
      penalty_t = max(eigs_t) / eigs_t;
      
      
      for(int i = 0; i < m; i++){
        
        Facload_sparse_perm_t.row(i) = savs(Fac_norm_perm_t, 
                                  Facload_perm_t.row(i), 2, penalty_t.t());
      }
      
      n_active_tmp1 = sum(abs(Facload_sparse_perm_t), 0);
      n_active_tmp1.elem(find(n_active_tmp1>0)).ones();
      n_active = accu(n_active_tmp1);
//      int n_active = 0;
//      for(int i = 0; i < r; i++){
//        if(n_active_tmp1(i)>0){
//          n_active++;
//        }
//      }
      if(t==t_ind(j)){
        Facload_sparse_t_store_tmp.slice(j) = Facload_sparse_perm_t.cols(reorder_t);
        if(j!=(n_t - 1)){
          j++;
        }
      }
      
      active_sparse_store(t, rep) = n_active;
    }
    Facload_sparse_t_store(rep) = Facload_sparse_t_store_tmp;
    p.increment();
  }
  List out = List::create(
    Named("eigenvalues") = eigs_store,
    Named("active") = active_sparse_store,
    Named("Facload_t") = Facload_sparse_t_store
  );
  
  return out;
}

