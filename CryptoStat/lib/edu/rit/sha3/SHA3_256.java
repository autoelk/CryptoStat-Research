//******************************************************************************
//
// File:    SHA3_256.java
// Package: edu.rit.sha3
// Unit:    Class edu.rit.sha3.SHA3_256
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

import edu.rit.util.BigInt;

/**
 * Class SHA3_256 implements the SHA3-256 hash function with a 64-byte (512-bit)
 * message. The SHA3-256 cryptographic function's input <I>A</I> is the first
 * portion of the message (256 bits); input <I>B</I> is the second portion of
 * the message (256 bits); output <I>C</I> is the digest (256 bits).
 *
 * @author  Aziel Shaw
 * @version 21-Mar-2019
 */
public class SHA3_256 extends SHA3Base {
    
    /**
     * Returns the constructor
     * @return The constructor for SHA3 256
     */
    @Override
    public String constructor() {
        return "edu.rit.sha3.SHA3_256()";
    }

    /**
     * Description of this hash function
     * @return Description of this hash function
     */
    @Override
    public String description() {
        return "SHA3-256 hash function";
    }

    /**
     * The bitsize of the digest
     * @return The bitsize of the digest
     */
    @Override
    public int C_bitSize() {
        return 256;
    }

    /**
     * The module location for the CUDA version of this hash function
     * @return The module location for the CUDA version of this hash function
     */
    @Override
    protected String moduleName() {
        return "edu/rit/sha3/SHA3_256.ptx";
    }
    
    /**
     * Instantiates the Rate of the hash function using the given A & B BigInts
     * @param A BigInt A
     * @param B BigInt B
     */
    @Override
    protected void instanciateRate(BigInt A, BigInt B) {
        // Get long arrays from A and B
        long[] Aarr = new long[4];
        A.unpackBigEndian(Aarr);
        long[] Barr = new long[4];
        B.unpackBigEndian(Barr);
        
        // Instanciate state variables
        A00 = Aarr[0]; // A1
        A10 = Aarr[1]; // A2
        A20 = Aarr[2]; // A3
        A30 = Aarr[3]; // A4
        A40 = Barr[0]; // B1
        A01 = Barr[1]; // B2
        A11 = Barr[2]; // B3
        A21 = Barr[3]; // B4
        // Padding
        A31 = 0x0000000000000006L; // PADDING 1
        A41 = 0x0000000000000000L; // PADDING 2
        A02 = 0x0000000000000000L; // PADDING 3
        A12 = 0x0000000000000000L; // PADDING 4
        A22 = 0x0000000000000000L; // PADDING 5
        A32 = 0x0000000000000000L; // PADDING 6
        A42 = 0x0000000000000000L; // PADDING 7
        A03 = 0x0000000000000000L; // PADDING 8
        A13 = 0x8000000000000000L; // PADDING 9
        // Capacity
        A23 = 0x0000000000000000L; // c1
        A33 = 0x0000000000000000L; // c2
        A43 = 0x0000000000000000L; // c3
        A04 = 0x0000000000000000L; // c4
        A14 = 0x0000000000000000L; // c5
        A24 = 0x0000000000000000L; // c6
        A34 = 0x0000000000000000L; // c7
        A44 = 0x0000000000000000L; // c8
    }

    /**
     * Records the Digest
     * @param dig The BigInt to record into
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
    @Override
    protected void recordDigest(BigInt dig, 
            long a, long b, long c, long d, long e, 
            long f, long g, long h, long i, long j, 
            long k, long l, long m, long n, long o, 
            long p, long q, long r, long s, long t, 
            long u, long v, long w, long x, long y) {
        long tmp;
        tmp = Long.reverseBytes(a);
        dig.value[7] = (int)(tmp >> 32);
        dig.value[6] = (int)(tmp);
        tmp = Long.reverseBytes(b);
        dig.value[5] = (int)(tmp >> 32);
        dig.value[4] = (int)(tmp);
        tmp = Long.reverseBytes(c);
        dig.value[3] = (int)(tmp >> 32);
        dig.value[2] = (int)(tmp);
        tmp = Long.reverseBytes(d);
        dig.value[1] = (int)(tmp >> 32);
        dig.value[0] = (int)(tmp);
    }
}
