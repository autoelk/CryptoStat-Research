//******************************************************************************
//
// File:    SHAKE128.java
// Package: edu.rit.sha3
// Unit:    Class edu.rit.sha3.SHAKE128
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
 * Class SHAKE128 implements the SHAKE-128 extensible output hash function with
 * a 64-byte (512-bit) message. The SHAKE-128 cryptographic function's input
 * <I>A</I> is the first portion of the message (256 bits); input <I>B</I> is
 * the second portion of the message (256 bits); output <I>C</I> is the digest
 * (fixed at 2048 bits).
 *
 * @author  Aziel Shaw
 * @version 21-Mar-2019
 */
public class SHAKE128 extends Function {
    
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
    
    private static int Csize = 2048;
    
    /**
     * Default Constructor
     */
    public SHAKE128() {
        
    }
    
    /**
     * Returns the constructor
     * @return The constructor for SHAKE128
     */
    @Override
    public String constructor() {
        return "edu.rit.sha3.SHAKE128()";
    }
    
    /**
     * The module location for the CUDA version of this hash function
     * @return The module location for the CUDA version of this hash function
     */
    @Override
    protected String moduleName() {
        return "edu/rit/sha3/SHAKE128.ptx";
    }

    /**
     * Description of this hash function
     * @return Description of this hash function
     */
    @Override
    public String description() {
        return "SHAKE128 XOR hash function";
    }

    @Override
    public String A_description() {
        return "message bits 0-255";
    }

    @Override
    public int A_bitSize() {
        return 256;
    }

    @Override
    public String B_description() {
        return "message bits 256-511";
    }

    @Override
    public int B_bitSize() {
        return 256;
    }

    @Override
    public String C_description() {
        return "digest";
    }

    @Override
    public int C_bitSize() {
        return this.Csize;
    }

    @Override
    public int rounds() {
        return 24;
    }

    // State variables
    long A00, A01, A02, A03, A04,
         A10, A11, A12, A13, A14,
         A20, A21, A22, A23, A24,
         A30, A31, A32, A33, A34,
         A40, A41, A42, A43, A44;
    
    @Override
    public void evaluate(BigInt A, BigInt B, BigInt[] C) {
        // Get long arrays from A and B
        long[] Aarr = new long[4];
        A.unpackBigEndian(Aarr);
        long[] Barr = new long[4];
        B.unpackBigEndian(Barr);
        
        // Instanciate state variables
        A00 = Long.reverseBytes(Aarr[0]); // A1
        A10 = Long.reverseBytes(Aarr[1]); // A2
        A20 = Long.reverseBytes(Aarr[2]); // A3
        A30 = Long.reverseBytes(Aarr[3]); // A4
        A40 = Long.reverseBytes(Barr[0]); // B1
        A01 = Long.reverseBytes(Barr[1]); // B2
        A11 = Long.reverseBytes(Barr[2]); // B3
        A21 = Long.reverseBytes(Barr[3]); // B4
        // PADDING
        A31 = 0x000000000000001fL;
        A41 = 0x0000000000000000L;
        A02 = 0x0000000000000000L;
        A12 = 0x0000000000000000L;
        A22 = 0x0000000000000000L;
        A32 = 0x0000000000000000L;
        A42 = 0x0000000000000000L;
        A03 = 0x0000000000000000L;
        A13 = 0x0000000000000000L;
        A23 = 0x0000000000000000L;
        A33 = 0x0000000000000000L;
        A43 = 0x0000000000000000L;
        A04 = 0x8000000000000000L;
        // CAPACITY
        A14 = 0x0000000000000000L;
        A24 = 0x0000000000000000L;
        A34 = 0x0000000000000000L;
        A44 = 0x0000000000000000L;
        
        // Local temporary variables.
        long B00, B01, B02, B03, B04;
        long B10, B11, B12, B13, B14;
        long B20, B21, B22, B23, B24;
        long B30, B31, B32, B33, B34;
        long B40, B41, B42, B43, B44;
        long C0, C1, C2, C3, C4;
        long D0, D1, D2, D3, D4;
        
        // Iterate over 24 rounds
        for (int roundNum = 0; roundNum < 24; roundNum++) {
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
            recordDigest(C[roundNum], A00, 63);
            recordDigest(C[roundNum], A10, 61);
            recordDigest(C[roundNum], A20, 59);
            recordDigest(C[roundNum], A30, 57);
            recordDigest(C[roundNum], A40, 55);
            recordDigest(C[roundNum], A01, 53);
            recordDigest(C[roundNum], A11, 51);
            recordDigest(C[roundNum], A21, 49);
            recordDigest(C[roundNum], A31, 47);
            recordDigest(C[roundNum], A41, 45);
            recordDigest(C[roundNum], A02, 43);
            recordDigest(C[roundNum], A12, 41);
            recordDigest(C[roundNum], A22, 39);
            recordDigest(C[roundNum], A32, 37);
            recordDigest(C[roundNum], A42, 35);
            recordDigest(C[roundNum], A03, 33);
            recordDigest(C[roundNum], A13, 31);
            recordDigest(C[roundNum], A23, 29);
            recordDigest(C[roundNum], A33, 27);
            recordDigest(C[roundNum], A43, 25);
            recordDigest(C[roundNum], A04, 23);
            if(roundNum != 23) {
                // EXTRA ROUND
                long AA00, AA01, AA02, AA03, AA04,
                     AA10, AA11, AA12, AA13, AA14,
                     AA20, AA21, AA22, AA23, AA24,
                     AA30, AA31, AA32, AA33, AA34,
                     AA40, AA41, AA42, AA43, AA44;
                AA00 = A00;
                AA01 = A01;
                AA02 = A02;
                AA03 = A03;
                AA04 = A04;
                AA10 = A10;
                AA11 = A11;
                AA12 = A12;
                AA13 = A13;
                AA14 = A14;
                AA20 = A20;
                AA21 = A21;
                AA22 = A22;
                AA23 = A23;
                AA24 = A24;
                AA30 = A30;
                AA31 = A31;
                AA32 = A32;
                AA33 = A33;
                AA34 = A34;
                AA40 = A40;
                AA41 = A41;
                AA42 = A42;
                AA43 = A43;
                AA44 = A44;
                // Theta step mapping
                // xor A lanes
                C0 = AA00 ^ AA01 ^ AA02 ^ AA03 ^ AA04;
                C1 = AA10 ^ AA11 ^ AA12 ^ AA13 ^ AA14;
                C2 = AA20 ^ AA21 ^ AA22 ^ AA23 ^ AA24;
                C3 = AA30 ^ AA31 ^ AA32 ^ AA33 ^ AA34;
                C4 = AA40 ^ AA41 ^ AA42 ^ AA43 ^ AA44;
                // xor and rotate
                D0 = C3 ^ Long.rotateLeft(C0, 1);
                D1 = C4 ^ Long.rotateLeft(C1, 1);
                D2 = C0 ^ Long.rotateLeft(C2, 1);
                D3 = C1 ^ Long.rotateLeft(C3, 1);
                D4 = C2 ^ Long.rotateLeft(C4, 1);
                // More lane xoring
                AA00 ^= D1;
                AA01 ^= D1;
                AA02 ^= D1;
                AA03 ^= D1;
                AA04 ^= D1;
                AA10 ^= D2;
                AA11 ^= D2;
                AA12 ^= D2;
                AA13 ^= D2;
                AA14 ^= D2;
                AA20 ^= D3;
                AA21 ^= D3;
                AA22 ^= D3;
                AA23 ^= D3;
                AA24 ^= D3;
                AA30 ^= D4;
                AA31 ^= D4;
                AA32 ^= D4;
                AA33 ^= D4;
                AA34 ^= D4;
                AA40 ^= D0;
                AA41 ^= D0;
                AA42 ^= D0;
                AA43 ^= D0;
                AA44 ^= D0;

                // rho step mapping & pi step mapping
                B00 = AA00;
                B13 = Long.rotateLeft(AA01, 36);
                B21 = Long.rotateLeft(AA02, 3);
                B34 = Long.rotateLeft(AA03, 41);
                B42 = Long.rotateLeft(AA04, 18);
                B02 = Long.rotateLeft(AA10, 1);
                B10 = Long.rotateLeft(AA11, 44);
                B23 = Long.rotateLeft(AA12, 10);
                B31 = Long.rotateLeft(AA13, 45);
                B44 = Long.rotateLeft(AA14, 2);
                B04 = Long.rotateLeft(AA20, 62);
                B12 = Long.rotateLeft(AA21, 6);
                B20 = Long.rotateLeft(AA22, 43);
                B33 = Long.rotateLeft(AA23, 15);
                B41 = Long.rotateLeft(AA24, 61);
                B01 = Long.rotateLeft(AA30, 28);
                B14 = Long.rotateLeft(AA31, 55);
                B22 = Long.rotateLeft(AA32, 25);
                B30 = Long.rotateLeft(AA33, 21);
                B43 = Long.rotateLeft(AA34, 56);
                B03 = Long.rotateLeft(AA40, 27);
                B11 = Long.rotateLeft(AA41, 20);
                B24 = Long.rotateLeft(AA42, 39);
                B32 = Long.rotateLeft(AA43, 8);
                B40 = Long.rotateLeft(AA44, 14);

                // chi step mapping
                AA00 = B00 ^ (~B10 & B20);
                AA01 = B01 ^ (~B11 & B21);
                AA02 = B02 ^ (~B12 & B22);
                AA03 = B03 ^ (~B13 & B23);
                AA04 = B04 ^ (~B14 & B24);
                AA10 = B10 ^ (~B20 & B30);
                AA11 = B11 ^ (~B21 & B31);
                AA12 = B12 ^ (~B22 & B32);
                AA13 = B13 ^ (~B23 & B33);
                AA14 = B14 ^ (~B24 & B34);
                AA20 = B20 ^ (~B30 & B40);
                AA21 = B21 ^ (~B31 & B41);
                AA22 = B22 ^ (~B32 & B42);
                AA23 = B23 ^ (~B33 & B43);
                AA24 = B24 ^ (~B34 & B44);
                AA30 = B30 ^ (~B40 & B00);
                AA31 = B31 ^ (~B41 & B01);
                AA32 = B32 ^ (~B42 & B02);
                AA33 = B33 ^ (~B43 & B03);
                AA34 = B34 ^ (~B44 & B04);
                AA40 = B40 ^ (~B00 & B10);
                AA41 = B41 ^ (~B01 & B11);
                AA42 = B42 ^ (~B02 & B12);
                AA43 = B43 ^ (~B03 & B13);
                AA44 = B44 ^ (~B04 & B14);
                // iota step mapping
                AA00 = AA00 ^ K[roundNum];
                // Saving resultant array
                recordDigest(C[roundNum], AA00, 21);
                recordDigest(C[roundNum], AA10, 19);
                recordDigest(C[roundNum], AA20, 17);
                recordDigest(C[roundNum], AA30, 15);
                recordDigest(C[roundNum], AA40, 13);
                recordDigest(C[roundNum], AA01, 11);
                recordDigest(C[roundNum], AA11,  9);
                recordDigest(C[roundNum], AA21,  7);
                recordDigest(C[roundNum], AA31,  5);
                recordDigest(C[roundNum], AA41,  3);
                recordDigest(C[roundNum], AA02,  1);
            }
        } // End round for loop
        for (int roundNum = 0; roundNum < 24; roundNum++) {
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
        } // End round for loop
        recordDigest(C[23], A00, 21);
        recordDigest(C[23], A10, 19);
        recordDigest(C[23], A20, 17);
        recordDigest(C[23], A30, 15);
        recordDigest(C[23], A40, 13);
        recordDigest(C[23], A01, 11);
        recordDigest(C[23], A11,  9);
        recordDigest(C[23], A21,  7);
        recordDigest(C[23], A31,  5);
        recordDigest(C[23], A41,  3);
        recordDigest(C[23], A02,  1);
    }

    /**
     * Records the Digest
     * @param dig The BigInt to record into
     * @param num Number to add to Digest
     */
    protected void recordDigest(BigInt dig, long num, int indexNum) {
        long tmp;
        tmp = Long.reverseBytes(num);
        dig.value[indexNum    ] = (int)(tmp >> 32);
        dig.value[indexNum - 1] = (int)(tmp);
    }
    
}
