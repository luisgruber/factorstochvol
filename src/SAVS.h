/*
 * R package stochvol by
 *     Gregor Kastner Copyright (C) 2016-2021
 *     Darjus Hosszejni Copyright (C) 2019-2021
 *     Luis Gruber Copyright (C) 2021
 *
 *  This file is part of the R package factorstochvol: Bayesian Estimation
 *  of (Sparse) Latent Factor Stochastic Volatility Models
 *
 *  The R package factorstochvol is free software: you can redistribute
 *  it and/or modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation, either version 2 or any
 *  later version of the License.
 *
 *  The R package factorstochvol is distributed in the hope that it will
 *  be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 *  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with the R package factorstochvol. If that is not the case,
 *  please refer to <http://www.gnu.org/licenses/>.
 */

#ifndef _SAVS_H
#define _SAVS_H

//#define ARMA_NO_DEBUG // disables bounds checks
#include <RcppArmadillo.h>

RcppExport SEXP DSAVS(const SEXP, const SEXP t_to_store);
RcppExport SEXP DSAVS2(const SEXP fsvdraws_in, const SEXP each_in,
                       const SEXP store_all_in, const SEXP type_in,
                       const SEXP nu_in);
RcppExport SEXP DSAVS3(const SEXP Facload_draws_in,
                       const SEXP Fac_draws_in,
                       const SEXP logvar_draws_in,
                       const SEXP each_in,
                       const SEXP store_all_in, const SEXP type_in,
                       const SEXP nu_in);
void facload_sparse(arma::mat& Facload_sparse_t,
                    arma::vec& penalty_t,
                    const arma::mat& Fac,
                    const arma::mat& Faclogvar,
                    const arma::vec& Idilogvar,
                    const arma::mat& Facload,
                    const arma::rowvec& Facvol_t,
                    const arma::uvec& match_ind,
                    const std::string& type,
                    const double& nu,
                    const arma::vec& type_vec);
RcppExport SEXP predsavs_cpp(const SEXP fsvdraws_in, const SEXP ahead_in,
                             const SEXP penalty_type_in,
                             const SEXP nu_in,
                             const SEXP type_in);
RcppExport SEXP predsavs_cpp_beta(const SEXP fsvdraws_in,
                                  const SEXP ahead_in,
                                  const SEXP penalty_type_in,
                                  const SEXP nu_in,
                                  const SEXP type_in);

#endif
