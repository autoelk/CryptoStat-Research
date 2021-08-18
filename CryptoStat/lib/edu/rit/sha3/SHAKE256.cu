//******************************************************************************
//
// File:    SHAKE256.cu
// Unit:    SHAKE-256 CUDA functions
// Author:  Aziel Shaw
//
// This CUDA source file is copyright (C) 2019 by Alan Kaminsky. All rights
// reserved. For further information, contact Alan Kaminsky at ark@cs.rit.edu.
//
// This CUDA source file is part of the CryptoStat Library ("CryptoStat").
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

#ifndef __SHAKE256_CU__
#define __SHAKE256_CU__

#include <stdint.h>
#include "../util/BigInt.cu"

// Round constants.
__constant__ uint64_t K [24] = {
    0x0000000000000001L, 0x0000000000008082L, 
    0x800000000000808AL, 0x8000000080008000L, 
    0x000000000000808BL, 0x0000000080000001L,
    0x8000000080008081L, 0x8000000000008009L, 
    0x000000000000008AL, 0x0000000000000088L, 
    0x0000000080008009L, 0x000000008000000AL,
    0x000000008000808BL, 0x800000000000008BL, 
    0x8000000000008089L, 0x8000000000008003L, 
    0x8000000000008002L, 0x8000000000000080L,
    0x000000000000800AL, 0x800000008000000AL, 
    0x8000000080008081L, 0x8000000000008080L, 
    0x0000000080000001L, 0x8000000080008008L
};

/**
 * Left-rotate functions.
 */
__device__ uint64_t ROTL_1 (uint64_t x) {
	return (x << 1) | (x >> 63);
}

__device__ uint64_t ROTL_2 (uint64_t x) {
	return (x << 2) | (x >> 62);
}

__device__ uint64_t ROTL_3 (uint64_t x) {
	return (x << 3) | (x >> 61);
}

__device__ uint64_t ROTL_6 (uint64_t x) {
	return (x << 6) | (x >> 58);
}

__device__ uint64_t ROTL_8 (uint64_t x) {
	return (x << 8) | (x >> 56);
}

__device__ uint64_t ROTL_10 (uint64_t x) {
	return (x << 10) | (x >> 54);
}

__device__ uint64_t ROTL_14 (uint64_t x) {
	return (x << 14) | (x >> 50);
}

__device__ uint64_t ROTL_15 (uint64_t x) {
	return (x << 15) | (x >> 49);
}

__device__ uint64_t ROTL_18 (uint64_t x) {
	return (x << 18) | (x >> 46);
}

__device__ uint64_t ROTL_20 (uint64_t x) {
	return (x << 20) | (x >> 44);
}

__device__ uint64_t ROTL_21 (uint64_t x) {
	return (x << 21) | (x >> 43);
}

__device__ uint64_t ROTL_25 (uint64_t x) {
	return (x << 25) | (x >> 39);
}

__device__ uint64_t ROTL_27 (uint64_t x) {
	return (x << 27) | (x >> 37);
}

__device__ uint64_t ROTL_28 (uint64_t x) {
	return (x << 28) | (x >> 36);
}

__device__ uint64_t ROTL_36 (uint64_t x) {
	return (x << 36) | (x >> 28);
}

__device__ uint64_t ROTL_39 (uint64_t x) {
	return (x << 39) | (x >> 25);
}

__device__ uint64_t ROTL_41 (uint64_t x) {
	return (x << 41) | (x >> 23);
}

__device__ uint64_t ROTL_43 (uint64_t x) {
	return (x << 43) | (x >> 21);
}

__device__ uint64_t ROTL_44 (uint64_t x) {
	return (x << 44) | (x >> 20);
}

__device__ uint64_t ROTL_45 (uint64_t x) {
	return (x << 45) | (x >> 19);
}

__device__ uint64_t ROTL_55 (uint64_t x) {
	return (x << 55) | (x >> 9);
}

__device__ uint64_t ROTL_56 (uint64_t x) {
	return (x << 56) | (x >> 8);
}

__device__ uint64_t ROTL_61 (uint64_t x) {
	return (x << 61) | (x >> 3);
}

__device__ uint64_t ROTL_62 (uint64_t x) {
	return (x << 62) | (x >> 2);
}

/**
 * Reverse the bytes of the given long
 */
__device__ uint64_t reverseBytes(uint64_t x) {
	return ((((x) >> 56) & 0x00000000000000FF) | 
		(((x) >> 40) & 0x000000000000FF00) |
		(((x) >> 24) & 0x0000000000FF0000) | 
		(((x) >>  8) & 0x00000000FF000000) |
		(((x) <<  8) & 0x000000FF00000000) | 
		(((x) << 24) & 0x0000FF0000000000) |
		(((x) << 40) & 0x00FF000000000000) | 
		(((x) << 56) & 0xFF00000000000000));
}

/**
 * Record digest for the current round. Working variables a through h are
 * stored in the proper words of the digest (dig).
 */
 __device__ void recordDigest (uint32_t* dig, uint64_t val, int index) {
    uint64_t tmp;
    tmp = reverseBytes(val);
    dig[index    ] = (uint32_t)(tmp >> 32);
    dig[index - 1] = (uint32_t)(tmp);
}

/**
 * Evaluate the cryptographic function SHA3
 */
__device__ void evaluate(int NA, int Asize, uint32_t* A,
 	                     int NB, int Bsize, uint32_t* B,
	                     int R, int Csize, uint32_t* C,
	                     int a, int b) {
	// State variables
	uint64_t W [25];
	// Instanciate the rate of the SHA3 state
		// Get data from A & B
	biUnpackLongBigEndian (Asize, A, W, 0);
	biUnpackLongBigEndian (Bsize, B, W, 4);
		// Fill rest with padding and capacity
	W[ 0] = reverseBytes(W[ 0]);
	W[ 1] = reverseBytes(W[ 1]);
	W[ 2] = reverseBytes(W[ 2]);
	W[ 3] = reverseBytes(W[ 3]);
	W[ 4] = reverseBytes(W[ 4]);
	W[ 5] = reverseBytes(W[ 5]);
	W[ 6] = reverseBytes(W[ 6]);
	W[ 7] = reverseBytes(W[ 7]);
    // Padding
    W[ 8] = 0x000000000000001fL;
    W[ 9] = 0x0000000000000000L;
    W[10] = 0x0000000000000000L;
    W[11] = 0x0000000000000000L;
    W[12] = 0x0000000000000000L;
    W[13] = 0x0000000000000000L;
    W[14] = 0x0000000000000000L;
    W[15] = 0x0000000000000000L;
    W[16] = 0x8000000000000000L;
    // Capacity
    W[17] = 0x0000000000000000L;
    W[18] = 0x0000000000000000L;
    W[19] = 0x0000000000000000L;
    W[20] = 0x0000000000000000L;
    W[21] = 0x0000000000000000L;
    W[22] = 0x0000000000000000L;
	W[23] = 0x0000000000000000L;
    W[24] = 0x0000000000000000L;

	// Local temporary variables.
	uint64_t A00, A01, A02, A03, A04;
	uint64_t A10, A11, A12, A13, A14;
	uint64_t A20, A21, A22, A23, A24;
	uint64_t A30, A31, A32, A33, A34;
	uint64_t A40, A41, A42, A43, A44;
	
	uint64_t B00, B01, B02, B03, B04;
	uint64_t B10, B11, B12, B13, B14;
	uint64_t B20, B21, B22, B23, B24;
	uint64_t B30, B31, B32, B33, B34;
	uint64_t B40, B41, B42, B43, B44;
	uint64_t C0, C1, C2, C3, C4;
	uint64_t D0, D1, D2, D3, D4;

	A00 = W[ 0];
	A10 = W[ 1];
	A20 = W[ 2];
	A30 = W[ 3];
	A40 = W[ 4];
	A01 = W[ 5];
	A11 = W[ 6];
	A21 = W[ 7];
	A31 = W[ 8];
	A41 = W[ 9];
	A02 = W[10];
	A12 = W[11];
	A22 = W[12];
	A32 = W[13];
	A42 = W[14];
	A03 = W[15];
	A13 = W[16];
	A23 = W[17];
	A33 = W[18];
	A43 = W[19];
	A04 = W[20];
	A14 = W[21];
	A24 = W[22];
	A34 = W[23];
	A44 = W[24];

	// Do 24 rounds.
	for (int roundNum = 0; roundNum < 24; roundNum++) {
		// Theta step mapping
		// xor A lanes
		C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04;
		C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14;
		C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24;
		C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34;
		C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44;
		// xor and rotate
		D0 = C3 ^ ROTL_1(C0);
		D1 = C4 ^ ROTL_1(C1);
		D2 = C0 ^ ROTL_1(C2);
		D3 = C1 ^ ROTL_1(C3);
		D4 = C2 ^ ROTL_1(C4);
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
		B13 = ROTL_36(A01);
		B21 = ROTL_3(A02);
		B34 = ROTL_41(A03);
		B42 = ROTL_18(A04);
		B02 = ROTL_1(A10);
		B10 = ROTL_44(A11);
		B23 = ROTL_10(A12);
		B31 = ROTL_45(A13);
		B44 = ROTL_2(A14);
		B04 = ROTL_62(A20);
		B12 = ROTL_6(A21);
		B20 = ROTL_43(A22);
		B33 = ROTL_15(A23);
		B41 = ROTL_61(A24);
		B01 = ROTL_28(A30);
		B14 = ROTL_55(A31);
		B22 = ROTL_25(A32);
		B30 = ROTL_21(A33);
		B43 = ROTL_56(A34);
		B03 = ROTL_27(A40);
		B11 = ROTL_20(A41);
		B24 = ROTL_39(A42);
		B32 = ROTL_8(A43);
		B40 = ROTL_14(A44);
		
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
		
		// Record digest for this round.
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A00, 63);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A10, 61);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A20, 59);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A30, 57);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A40, 55);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A01, 53);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A11, 51);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A21, 49);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A31, 47);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A41, 45);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A02, 43);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A12, 41);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A22, 39);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A32, 37);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A42, 35);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A03, 33);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A13, 31);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A23, 29);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A33, 27);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A43, 25);
		recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), A04, 23);
		if(roundNum != 23)  {
			// EXTRA ROUND
			uint64_t AA00, AA01, AA02, AA03, AA04,
			 	 	 AA10, AA11, AA12, AA13, AA14,
			 	 	 AA20, AA21, AA22, AA23, AA24,
			 	 	 AA30, AA31, AA32, AA33, AA34,
				 	 AA40, AA41, AA42, AA43, AA44;
			// Assign to local
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
			D0 = C3 ^ ROTL_1(C0);
			D1 = C4 ^ ROTL_1(C1);
			D2 = C0 ^ ROTL_1(C2);
			D3 = C1 ^ ROTL_1(C3);
			D4 = C2 ^ ROTL_1(C4);
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
			B13 = ROTL_36(AA01);
			B21 = ROTL_3(AA02);
			B34 = ROTL_41(AA03);
			B42 = ROTL_18(AA04);
			B02 = ROTL_1(AA10);
			B10 = ROTL_44(AA11);
			B23 = ROTL_10(AA12);
			B31 = ROTL_45(AA13);
			B44 = ROTL_2(AA14);
			B04 = ROTL_62(AA20);
			B12 = ROTL_6(AA21);
			B20 = ROTL_43(AA22);
			B33 = ROTL_15(AA23);
			B41 = ROTL_61(AA24);
			B01 = ROTL_28(AA30);
			B14 = ROTL_55(AA31);
			B22 = ROTL_25(AA32);
			B30 = ROTL_21(AA33);
			B43 = ROTL_56(AA34);
			B03 = ROTL_27(AA40);
			B11 = ROTL_20(AA41);
			B24 = ROTL_39(AA42);
			B32 = ROTL_8(AA43);
			B40 = ROTL_14(AA44);
			
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
			// Save
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA00, 21);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA10, 19);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA20, 17);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA30, 15);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA40, 13);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA01, 11);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA11,  9);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA21,  7);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA31,  5);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA41,  3);
			recordDigest(biElem3D(Csize, C, NB, R, a, b, roundNum), AA02,  1);
		}
	} // End round for loop
	// DO SHA3 AGAIN
	// Do 24 rounds.
	for (int roundNum = 0; roundNum < 24; roundNum++) {
		// Theta step mapping
		// xor A lanes
		C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04;
		C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14;
		C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24;
		C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34;
		C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44;
		// xor and rotate
		D0 = C3 ^ ROTL_1(C0);
		D1 = C4 ^ ROTL_1(C1);
		D2 = C0 ^ ROTL_1(C2);
		D3 = C1 ^ ROTL_1(C3);
		D4 = C2 ^ ROTL_1(C4);
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
		B13 = ROTL_36(A01);
		B21 = ROTL_3(A02);
		B34 = ROTL_41(A03);
		B42 = ROTL_18(A04);
		B02 = ROTL_1(A10);
		B10 = ROTL_44(A11);
		B23 = ROTL_10(A12);
		B31 = ROTL_45(A13);
		B44 = ROTL_2(A14);
		B04 = ROTL_62(A20);
		B12 = ROTL_6(A21);
		B20 = ROTL_43(A22);
		B33 = ROTL_15(A23);
		B41 = ROTL_61(A24);
		B01 = ROTL_28(A30);
		B14 = ROTL_55(A31);
		B22 = ROTL_25(A32);
		B30 = ROTL_21(A33);
		B43 = ROTL_56(A34);
		B03 = ROTL_27(A40);
		B11 = ROTL_20(A41);
		B24 = ROTL_39(A42);
		B32 = ROTL_8(A43);
		B40 = ROTL_14(A44);
		
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
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A00, 21);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A10, 19);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A20, 17);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A30, 15);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A40, 13);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A01, 11);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A11,  9);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A21,  7);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A31,  5);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A41,  3);
	recordDigest(biElem3D(Csize, C, NB, R, a, b, 23), A02,  1);
}

#include "../crst/FunctionKernel.cu"

#endif
