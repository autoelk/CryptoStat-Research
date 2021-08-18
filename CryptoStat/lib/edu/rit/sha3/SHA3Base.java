//******************************************************************************
//
// File:    SHA3Base.java
// Package: edu.rit.sha3
// Unit:    Class edu.rit.sha3.SHA3Base
//
// This Java source file is copyright (C) 2019 by Alan Kaminsky. All rights
// reserved. For further information, contact Alan Kaminsky at ark@cs.rit.edu.
//
// This Java source file is part of the CryptoStat Library ("CryptoStat").
// CryptoStat is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// CryptoStat is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.sha3;

import edu.rit.crst.Function;
import edu.rit.util.BigInt;

/**
 * Class SHA3Base is the abstract base class for the SHA-3 hash functions. It
 * implements a 64-byte (512-bit) message. The cryptographic function's input
 * <I>A</I> is the first portion of the message (256 bits); input <I>B</I> is
 * the second portion of the message (256 bits); output <I>C</I> is the digest
 * (size determined by a subclass).
 *
 * @author  Aziel Shaw
 * @version 21-Mar-2019
 */
public abstract class SHA3Base extends Function {

    /**
     * Round constants
     */
    private static final long[] K = new long[]{
        0x0000000000000001L, 0x0000000000008082L, 0x800000000000808AL,
        0x8000000080008000L, 0x000000000000808BL, 0x0000000080000001L,
        0x8000000080008081L, 0x8000000000008009L, 0x000000000000008AL,
        0x0000000000000088L, 0x0000000080008009L, 0x000000008000000AL,
        0x000000008000808BL, 0x800000000000008BL, 0x8000000000008089L,
        0x8000000000008003L, 0x8000000000008002L, 0x8000000000000080L,
        0x000000000000800AL, 0x800000008000000AL, 0x8000000080008081L,
        0x8000000000008080L, 0x0000000080000001L, 0x8000000080008008L
    };

    // State variables
    long A00, A01, A02, A03, A04,
         A10, A11, A12, A13, A14,
         A20, A21, A22, A23, A24,
         A30, A31, A32, A33, A34,
         A40, A41, A42, A43, A44;

    /**
     * Default constructor
     */
    public SHA3Base() {
        super();
    }

    /**
     * Returns the Description of the A value
     * @return The Description of the A value
     */
    @Override
    public String A_description() {
        return "message bits 0-255";
    }

    /**
     * The bitsize of A
     * @return The bitsize of A
     */
    @Override
    public int A_bitSize() {
        return 256;
    }

    /**
     * Returns the Description of the B value
     * @return The Description of the B value
     */
    @Override
    public String B_description() {
        return "message bits 256-511";
    }

    /**
     * The bitsize of B
     * @return The bitsize of B
     */
    @Override
    public int B_bitSize() {
        return 256;
    }

    /**
     * Returns the Description of C
     * @return The Description of C
     */
    @Override
    public String C_description() {
        return "digest";
    }

    /**
     * Returns the amount of rounds
     * @return Number of rounds
     */
    @Override
    public int rounds() {
        return 24;
    }

    /**
     * Records the Digest after each round of SHA3
     * @param dig 
     * @param a a
     * @param b b
     * @param c c
     * @param d d
     * @param e e
     * @param f f
     * @param g g
     * @param h h
     * @param i i
     * @param j j
     * @param k k
     * @param l l
     * @param m m
     * @param n n
     * @param o o
     * @param p p
     * @param q q
     * @param r r
     * @param s s
     * @param t t
     * @param u u
     * @param v v
     * @param w w
     * @param x x
     * @param y y
     */
    protected abstract void recordDigest(BigInt dig,
            long a, long b, long c, long d, long e,
            long f, long g, long h, long i, long j,
            long k, long l, long m, long n, long o,
            long p, long q, long r, long s, long t,
            long u, long v, long w, long x, long y);
    
    /**
     * Instanciate the rate given two BigInts
     * @param A BigInt A
     * @param B BigInt B
     */
    protected abstract void instanciateRate(BigInt A, BigInt B);

    /**
     * Evaluate the SHA3 function, given BigInts A, B, and an array of output
     * BigInts C
     * @param A BigInt A
     * @param B BigInt B
     * @param C BigInt array C
     */
    @Override
    public void evaluate(BigInt A, BigInt B, BigInt[] C) {
        
        // Instanciate state variables
        instanciateRate(A, B);

        // Local temporary variables.
        long B00, B01, B02, B03, B04;
        long B10, B11, B12, B13, B14;
        long B20, B21, B22, B23, B24;
        long B30, B31, B32, B33, B34;
        long B40, B41, B42, B43, B44;
        long C0, C1, C2, C3, C4;
        long D0, D1, D2, D3, D4;
        
        // Reverse the input
        A00 = Long.reverseBytes(A00);
	A10 = Long.reverseBytes(A10);
        A20 = Long.reverseBytes(A20);
        A30 = Long.reverseBytes(A30);
        A40 = Long.reverseBytes(A40);
        A01 = Long.reverseBytes(A01);
        A11 = Long.reverseBytes(A11);
        A21 = Long.reverseBytes(A21);

        // Iterate over 24 rounds
        for (int roundNum = 0; roundNum < rounds(); roundNum++) {
            // Theta step mapping
            // xor A lanes
            C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04;
            C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14;
            C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24;
            C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34;
            C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44;
            // xor and rotate
            D0 = C3 ^ Long.rotateLeft(C0, 1);
            D1 = C4 ^ Long.rotateLeft(C1, 1);
            D2 = C0 ^ Long.rotateLeft(C2, 1);
            D3 = C1 ^ Long.rotateLeft(C3, 1);
            D4 = C2 ^ Long.rotateLeft(C4, 1);
            // More lane xoring
            A00 ^= D1;
            A01 ^= D1;
            A02 ^= D1;
            A03 ^= D1;
            A04 ^= D1;
            A10 ^= D2;
            A11 ^= D2;
            A12 ^= D2;
            A13 ^= D2;
            A14 ^= D2;
            A20 ^= D3;
            A21 ^= D3;
            A22 ^= D3;
            A23 ^= D3;
            A24 ^= D3;
            A30 ^= D4;
            A31 ^= D4;
            A32 ^= D4;
            A33 ^= D4;
            A34 ^= D4;
            A40 ^= D0;
            A41 ^= D0;
            A42 ^= D0;
            A43 ^= D0;
            A44 ^= D0;
            
            // rho step mapping & pi step mapping
            B00 = A00;
            B13 = Long.rotateLeft(A01, 36);
            B21 = Long.rotateLeft(A02, 3);
            B34 = Long.rotateLeft(A03, 41);
            B42 = Long.rotateLeft(A04, 18);
            B02 = Long.rotateLeft(A10, 1);
            B10 = Long.rotateLeft(A11, 44);
            B23 = Long.rotateLeft(A12, 10);
            B31 = Long.rotateLeft(A13, 45);
            B44 = Long.rotateLeft(A14, 2);
            B04 = Long.rotateLeft(A20, 62);
            B12 = Long.rotateLeft(A21, 6);
            B20 = Long.rotateLeft(A22, 43);
            B33 = Long.rotateLeft(A23, 15);
            B41 = Long.rotateLeft(A24, 61);
            B01 = Long.rotateLeft(A30, 28);
            B14 = Long.rotateLeft(A31, 55);
            B22 = Long.rotateLeft(A32, 25);
            B30 = Long.rotateLeft(A33, 21);
            B43 = Long.rotateLeft(A34, 56);
            B03 = Long.rotateLeft(A40, 27);
            B11 = Long.rotateLeft(A41, 20);
            B24 = Long.rotateLeft(A42, 39);
            B32 = Long.rotateLeft(A43, 8);
            B40 = Long.rotateLeft(A44, 14);
            
            // chi step mapping
            A00 = B00 ^ (~B10 & B20);
            A01 = B01 ^ (~B11 & B21);
            A02 = B02 ^ (~B12 & B22);
            A03 = B03 ^ (~B13 & B23);
            A04 = B04 ^ (~B14 & B24);
            A10 = B10 ^ (~B20 & B30);
            A11 = B11 ^ (~B21 & B31);
            A12 = B12 ^ (~B22 & B32);
            A13 = B13 ^ (~B23 & B33);
            A14 = B14 ^ (~B24 & B34);
            A20 = B20 ^ (~B30 & B40);
            A21 = B21 ^ (~B31 & B41);
            A22 = B22 ^ (~B32 & B42);
            A23 = B23 ^ (~B33 & B43);
            A24 = B24 ^ (~B34 & B44);
            A30 = B30 ^ (~B40 & B00);
            A31 = B31 ^ (~B41 & B01);
            A32 = B32 ^ (~B42 & B02);
            A33 = B33 ^ (~B43 & B03);
            A34 = B34 ^ (~B44 & B04);
            A40 = B40 ^ (~B00 & B10);
            A41 = B41 ^ (~B01 & B11);
            A42 = B42 ^ (~B02 & B12);
            A43 = B43 ^ (~B03 & B13);
            A44 = B44 ^ (~B04 & B14);
            // iota step mapping
            A00 = A00 ^ K[roundNum];
            // Saving resultant array
            recordDigest(C[roundNum],
                A00, A10, A20, A30, A40,
                A01, A11, A21, A31, A41,
                A02, A12, A22, A32, A42,
                A03, A13, A23, A33, A43,
                A04, A14, A24, A34, A44);
        } // End round for loop
    }
}
