/*
 * Copyright (c) 2011 Peter Vasil
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * Neither the name of the project's author nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <QtCore/QCoreApplication>

//int main(int argc, char *argv[])
//{
//    QCoreApplication a(argc, argv);

//    return a.exec();
//    //    return 0;
//}

/*******************************************************
 A simple program that demonstrates the Eigen library.
 The program defines a random symmetric matrix
 and computes its eigendecomposition.
 For further details read the Eigen Reference Manual
********************************************************/

#include <stdlib.h>
#include <time.h>
#include <string.h>

// the following two are needed for printing
#include <iostream>
#include <iomanip>
/**************************************
/* The Eigen include files         */
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
/***************************************/

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {

    QCoreApplication a(argc, argv);

    int numVals = 9;
    double x[] = {1.0, 1.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0};
    double y[] = {1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0};
    double meanX, meanY;
    meanX = meanY = 0.0;
    double f1[] = {1.0, -0.5, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
    double f2[] = {1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 1.0};

    for(int i = 0; i < numVals; ++i)
    {
        meanX += x[i];
        meanY += y[i];
    }
    meanX /= (double)numVals;
    meanY /= (double)numVals;

    int M = 6, N = 1;
    MatrixXd Xres(M,M);
    MatrixXd xvec1(M,1);
    MatrixXd xvec2(M,1);

    double eps = 0.0001;

    for(int i = 0; i < numVals; ++i)
    {
        double xd = meanX-x[i];
        double yd = meanY-y[i];
        double d = sqrt(xd*xd+yd*yd);
        double td = 1.0 / (d*d + eps*eps);

        /* MatrixXd X(M,N); // Define an M x N general matrix */
        VectorXd X(M); // Define an M x N general matrix

        double xi = x[i] - meanX;
        double yi = y[i] - meanY;
        X(0) = 1.0;
        X(1) = xi;
        X(2) = yi;
        X(3) = xi*xi;
        X(4) = xi*yi;
        X(5) = yi*yi;

        MatrixXd C;
        C = td * X * X.transpose(); // fill in C by X * X^t.
        cout << "C" << endl;
        cout << C << endl;

        Xres += C;
        cout << "The symmetrix matrix Xres" << endl;
        cout << Xres << endl;

        xvec1 += td * X * f1[i];
        cout << "The symmetrix matrix xvec" << endl;
        cout << xvec1 << endl;
        xvec2 += td * X * f2[i];
        cout << "The symmetrix matrix xvec" << endl;
        cout << xvec2 << endl;
    }

    ColPivHouseholderQR<MatrixXd> qr(Xres); // decomposes C
    VectorXd s1 = qr.solve(xvec1);
    VectorXd s2 = qr.solve(xvec2);

    cout << "solution for right hand side r1" << endl;
    cout << s1 << endl;
    cout << "solution for right hand side r2" << endl;
    cout << s2 << endl;
    return 0;
}
