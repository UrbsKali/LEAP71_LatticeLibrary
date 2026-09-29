//
// SPDX-License-Identifier: Apache-2.0
//
// The LEAP 71 ShapeKernel is an open source geometry engine
// specifically for use in Computational Engineering Models (CEM).
//
// For more information, please visit https://leap71.com/shapekernel
// 
// This project is developed and maintained by LEAP 71 - © 2023 by LEAP 71
// https://leap71.com
//
// Computational Engineering will profoundly change our physical world in the
// years ahead. Thank you for being part of the journey.
//
// We have developed this library to be used widely, for both commercial and
// non-commercial projects alike. Therefore, have released it under a permissive
// open-source license.
// 
// The LEAP 71 ShapeKernel is based on the PicoGK compact computational geometry 
// framework. See https://picogk.org for more information.
//
// LEAP 71 licenses this file to you under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the
// License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, THE SOFTWARE IS
// PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//
// See the License for the specific language governing permissions and
// limitations under the License.
//

using System.Numerics;
using PicoGK;


namespace Leap71
{
    namespace LatticeLibrary
    {
        public class ImplicitHybridTpms : IImplicit
        {
            readonly IRawTPMSPattern m_patternA;
            readonly IRawTPMSPattern m_patternB;
            readonly float m_unitSize;
            readonly float m_wallThickness;
            readonly int m_axisIndex;
            readonly float m_center;
            readonly float m_sharpnessK;
            readonly bool m_useRange;
            readonly float m_start;
            readonly float m_end;

            /// <summary>
            /// Creates a hybrid TPMS using the paper-style sigmoid weighting:
            /// H(x,y,z) = a(x,y,z) * U1(x,y,z) + (1-a(x,y,z)) * U2(x,y,z)
            /// a(x,y,z) = 1 / (1 + exp(-K * G(x,y,z)))
            ///
            /// Here G is a simple plane function along an axis: G = coord(axis) - center.
            /// K controls transition sharpness (K > 0).
            ///
            /// NOTE: The TPMS raw fields are evaluated in normalized coordinates (pt / unitSize).
            /// The weight function uses the original (unscaled) point coordinates.
            /// </summary>
            public ImplicitHybridTpms(
                IRawTPMSPattern patternA,
                IRawTPMSPattern patternB,
                float unitSize,
                float wallThickness,
                int axisIndex,
                float center,
                float sharpnessK,
                bool useRange = false,
                float start = 0f,
                float end = 0f)
            {
                m_patternA = patternA;
                m_patternB = patternB;
                m_unitSize = unitSize;
                m_wallThickness = wallThickness;
                m_axisIndex = axisIndex;
                m_center = center;
                m_sharpnessK = sharpnessK;
                m_useRange = useRange;
                m_start = start;
                m_end = end;
            }

            public float fSignedDistance(in Vector3 vecPt)
            {
                float coord = m_axisIndex switch
                {
                    0 => vecPt.X,
                    1 => vecPt.Y,
                    _ => vecPt.Z
                };

                float a;
                if (m_useRange)
                {
                    float t = (coord - m_start) / (m_end - m_start);
                    t = float.Clamp(t, 0f, 1f);
                    a = SmoothStep(t);
                }
                else
                {
                    float g = coord - m_center;
                    a = Sigmoid(m_sharpnessK * g);
                }

                float x = vecPt.X / m_unitSize;
                float y = vecPt.Y / m_unitSize;
                float z = vecPt.Z / m_unitSize;

                float u1 = m_patternA.fGetSignedDistance(x, y, z);
                float u2 = m_patternB.fGetSignedDistance(x, y, z);

                float h = (a * u1) + ((1f - a) * u2);
                return MathF.Abs(h) - 0.5f * m_wallThickness;
            }

            static float Sigmoid(float t)
            {
                // Numerically stable logistic function.
                if (t >= 0f)
                {
                    float e = MathF.Exp(-t);
                    return 1f / (1f + e);
                }

                float eNeg = MathF.Exp(t);
                return eNeg / (1f + eNeg);
            }

            static float SmoothStep(float t)
            {
                // C1 continuous ramp 0..1
                return t * t * (3f - 2f * t);
            }
        }
    }
}
