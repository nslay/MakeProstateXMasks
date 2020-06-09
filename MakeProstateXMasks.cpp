/*-
 * Copyright (c) 2020 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include "Common.h"
#include "bsdgetopt.h"

// ITK stuff
#include "itkImageBase.h"
#include "itkPoint.h"

typedef std::mt19937_64 GeneratorType;

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-h] -f findings.csv -d t2wRootFolder -o outputRootFolder [-t trainingListFile.txt] [-v validationListFile.txt] [-r validationRatio] [-s seedString]" << std::endl;
  exit(1);
}

// Point, finding ID, Zone, Label
struct Finding {
  typedef itk::ImageBase<3>::PointValueType PointValueType;
  typedef itk::ImageBase<3>::PointType PointType;

  // Peripherap Zone
  // Transition Zone
  // Anterior fibromuscular Stroma
  // Seminal Vesicles
  enum ZoneType { UnknownZone = -1, PZ, TZ, AS, SV };
  enum LabelType { UnknownLabel = -1, False, True };

  std::string strPatientId;
  int iFindingId = -1;
  PointType clPosition;
  ZoneType eZone = UnknownZone;
  LabelType eLabel = UnknownLabel;

  bool SetZone(const std::string &strZone) {
    if (strZone == "PZ")
      eZone = PZ;
    else if (strZone == "TZ")
      eZone = TZ;
    else if (strZone == "AS")
      eZone = AS;
    else if (strZone == "SV")
      eZone = SV;
    else
      eZone = UnknownZone;

    return eZone != UnknownZone;
  }

  bool SetLabel(const std::string &strLabel) {
    if (strLabel == "TRUE")
      eLabel = True;
    else if (strLabel == "FALSE")
      eLabel = False;
    else
      eLabel = UnknownLabel;

    return eLabel != UnknownLabel;
  }

  double Distance2(const PointType &clOther) const { return (double)clPosition.SquaredEuclideanDistanceTo(clOther); }
  double Distance(const PointType &clOther) const { return (double)clPosition.EuclideanDistanceTo(clOther); }
};

GeneratorType & GetGenerator() {
  static GeneratorType clGenerator;
  return clGenerator;
}

std::vector<Finding> LoadFindings(const std::string &strFileName);
std::map<std::string, std::vector<Finding>> LoadFindingsMap(const std::string &strFileName);
std::unordered_map<std::string, std::vector<Finding>> LoadFindingsUnorderedMap(const std::string &strFileName);

// Make sure we fairly assign positive/negative cases to training/validation
// Returns index to end of training set [training|validation]
size_t BalancedShuffle(std::vector<std::pair<std::string, std::vector<Finding>>> &vFindingsPairs, double dValidationRatio);

template<typename PixelType>
typename itk::Image<PixelType, 3>::Pointer LoadT2WImage(const std::string &strPath);

// Returns z coordinates with positive examples
template<typename PixelType>
std::vector<itk::IndexValueType> MakeMask(itk::Image<PixelType, 3> *p_clMask, const std::vector<Finding> &vFindings, unsigned int a_iCounts[2]);

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  std::string strSeedString = "prostatex";
  std::string strDataRoot;
  std::string strOutputRoot;
  std::string strCsvFile;
  std::string strTrainingListFile = "trainingList.txt";
  std::string strValidationListFile = "validationList.txt";
  double dValidationRatio = 0.1; // 10% of the cases are hold out

  int c = 0;
  while ((c = getopt(argc, argv, "d:f:ho:r:s:t:v:")) != -1) {
    switch (c) {
    case 'd':
      strDataRoot = optarg;
      break;
    case 'f':
      strCsvFile = optarg;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'o':
      strOutputRoot = optarg;
      break;
    case 'r':
      {
        char *p = nullptr;
        dValidationRatio = strtod(optarg, &p);

        if (*p != '\0' || dValidationRatio < 0.0 || dValidationRatio > 1.0)
          Usage(p_cArg0);
      }
      break;
    case 's':
      strSeedString = optarg;
      break;
    case 't':
      strTrainingListFile = optarg;
      break;
    case 'v':
      strValidationListFile = optarg;
      break;
    case '?':
    default:
      Usage(p_cArg0);
      break;
    }
  }

  argc -= optind;
  argv += optind;

  if (strDataRoot.empty() || strOutputRoot.empty() || strCsvFile.empty())
    Usage(p_cArg0);

  std::cout << "Info: Using seed string '" << strSeedString << "'." << std::endl;

  std::seed_seq clSeed(strSeedString.begin(), strSeedString.end());
  GetGenerator().seed(clSeed);

  auto mFindings = LoadFindingsMap(strCsvFile);

  if (mFindings.empty()) {
    std::cerr << "Error: Failed to load findings." << std::endl;
    return -1;
  }

  // Make it a vector so we can shuffle
  std::vector<std::pair<std::string, std::vector<Finding>>> vFindingsPairs(mFindings.begin(), mFindings.end());

  std::cout << "Info: Shuffling findings ..." << std::endl;
  const size_t trainingSize = BalancedShuffle(vFindingsPairs, dValidationRatio);

  std::cout << "Info: Saving training list file to '" << strTrainingListFile << "' ..." << std::endl;
  std::cout << "Info: Saving validation list file to '" << strValidationListFile << "' (ratio = " << dValidationRatio << ") ..." << std::endl;

  std::ofstream trainingListStream(strTrainingListFile, std::ofstream::trunc);
  std::ofstream validationListStream(strValidationListFile, std::ofstream::trunc);

  if (!trainingListStream) {
    std::cerr << "Error: Could not open training list file." << std::endl;
    return -1;
  }

  if (!validationListStream) {
    std::cerr << "Error: Could not open validation list file." << std::endl;
    return -1;
  }

  typedef itk::Image<short, 3> ImageType;
  unsigned int a_uiTrainingCounts[2] = { 0, 0 };
  unsigned int a_uiValidationCounts[2] = { 0, 0 };

  size_t i = 0;
  for (const auto &stPair : vFindingsPairs) {
    std::cout << "Info: Processing '" << stPair.first << "' ..." << std::endl;

    const std::string strT2WPath = strDataRoot + '/' + stPair.first;

    ImageType::Pointer p_clT2WImage = LoadT2WImage<ImageType::PixelType>(strT2WPath);

    if (!p_clT2WImage) {
      std::cerr << "Error: Failed to load T2W image '" << strT2WPath << "'." << std::endl; 
      return -1;
    }

    ImageType::Pointer p_clMask = ImageType::New();

    p_clMask->SetRegions(p_clT2WImage->GetBufferedRegion());

    try {
      p_clMask->Allocate();
      p_clMask->FillBuffer(0);
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return -1;
    }

    p_clMask->SetOrigin(p_clT2WImage->GetOrigin());
    p_clMask->SetDirection(p_clT2WImage->GetDirection());
    p_clMask->SetSpacing(p_clT2WImage->GetSpacing());

    p_clMask->SetMetaDataDictionary(p_clT2WImage->GetMetaDataDictionary());

    // Set some DICOM information
    EncapsulateStringMetaData(p_clMask->GetMetaDataDictionary(), "0008|103e", std::string("Ground truth mask"));

    int iSeriesNumber = 0;
    if (ExposeStringMetaData(p_clT2WImage->GetMetaDataDictionary(), "0020|0011", iSeriesNumber))
      EncapsulateStringMetaData(p_clMask->GetMetaDataDictionary(), "0020|0011", iSeriesNumber + 10000);

    std::string strStudyDate;
    ExposeStringMetaData(p_clT2WImage->GetMetaDataDictionary(), "0008|0020", strStudyDate);

    unsigned int * const a_uiCounts = i < trainingSize ? a_uiTrainingCounts : a_uiValidationCounts;

    std::vector<itk::IndexValueType> vSlices = MakeMask<ImageType::PixelType>(p_clMask, stPair.second, a_uiCounts);

    if (vSlices.empty()) {
      std::cout << "Info: Negative case. Skipping..." << std::endl;
      continue;
    }

    const std::string strOutputPath = strOutputRoot + '/' + stPair.first + '/' + strStudyDate;

    std::cout << "Info: Saving mask to '" << strOutputPath << "' ..." << std::endl;

    if (!SaveDicomImage<ImageType::PixelType, 3>(p_clMask, strOutputPath, true)) {
      std::cerr << "Error: Failed to save DICOM series." << std::endl;
      return -1;
    }

    std::ostream &listStream = (i++) < trainingSize ? trainingListStream : validationListStream;

    for (auto index : vSlices) {
      // NOTE: Common's SaveDicomImage uses z+1 as index number
      listStream << strOutputPath << '/' << index+1 << ".dcm\n";
    }
  }

  std::cout << "Info: Training negative count = " << a_uiTrainingCounts[0] << ", positive count = " << a_uiTrainingCounts[1] << std::endl;
  std::cout << "Info: Validation negative count = " << a_uiValidationCounts[0] << ", positive count = " << a_uiValidationCounts[1] << std::endl;

  return 0;
}

std::vector<Finding> LoadFindings(const std::string &strFileName) {
  std::ifstream csvStream(strFileName);

  if (!csvStream) {
    std::cerr << "Error: Failed to open '" << strFileName << "'." << std::endl;
    return std::vector<Finding>();
  }

  // First line is a header
  std::string strLine;
  if (!std::getline(csvStream, strLine))
    return std::vector<Finding>();

  Trim(strLine);
  std::vector<std::string> vFields = SplitString<std::string>(strLine, ",");

  const size_t numFields = vFields.size();

  if (numFields != 4 && numFields != 5) {
    std::cerr << "Error: Expected 4 or 5 comma (',') delimited fields (got " << numFields << " fields)." << std::endl;
    return std::vector<Finding>();
  }

  Finding stFinding;
  std::vector<Finding> vFindings;
  std::vector<Finding::PointValueType> vPoint;

  while (std::getline(csvStream, strLine)) {
    Trim(strLine);

    if (strLine.empty()) // Empty line?
      continue;

    vFields = SplitString<std::string>(strLine, ",");

    if (vFields.size() != numFields) {
      std::cerr << "Error: Unexpected number of fields in '" << strLine << "' (expected " << numFields << " but got " << vFields.size() << ")." << std::endl;
      return std::vector<Finding>();
    }

    stFinding.strPatientId = vFields[0];

    {
      char *p = nullptr;
      stFinding.iFindingId = strtol(vFields[1].c_str(), &p, 10);

      if (*p != '\0') {
        std::cerr << "Error: Failed to parse finding ID '" << vFields[1] << "'." << std::endl;
        return std::vector<Finding>();
      }
    }

    Trim(vFields[2]);

    vPoint = SplitString<Finding::PointValueType>(vFields[2], " \t");

    if (vPoint.size() != 3) {
      std::cerr << "Error: Failed to parse 3D position '" << vFields[2] << "'." << std::endl;
      return std::vector<Finding>();
    }

    std::copy(vPoint.begin(), vPoint.end(), stFinding.clPosition.Begin());
    
    if (!stFinding.SetZone(vFields[3])) {
      std::cerr << "Error: Failed to set zone '" << vFields[3] << "'." << std::endl;
      return std::vector<Finding>();
    }

    if (numFields > 4 && !stFinding.SetLabel(vFields[4])) {
      std::cerr << "Error: Failed to set label '" << vFields[4] << "'." << std::endl;
      return std::vector<Finding>();
    }
    
    vFindings.emplace_back(std::move(stFinding));
  }

  return vFindings;
}

std::map<std::string, std::vector<Finding>> LoadFindingsMap(const std::string &strFileName) {
  std::vector<Finding> vFindings = LoadFindings(strFileName);

  std::map<std::string, std::vector<Finding>> mFindings;

  for (Finding &stFinding : vFindings)
    mFindings[stFinding.strPatientId].emplace_back(std::move(stFinding));

  return mFindings;
}

std::unordered_map<std::string, std::vector<Finding>> LoadFindingsUnorderedMap(const std::string &strFileName) {
  std::vector<Finding> vFindings = LoadFindings(strFileName);

  std::unordered_map<std::string, std::vector<Finding>> mFindings;

  for (Finding &stFinding : vFindings)
    mFindings[stFinding.strPatientId].emplace_back(std::move(stFinding));

  return mFindings;
}

// Returns index to end of training set [training|validation]
size_t BalancedShuffle(std::vector<std::pair<std::string, std::vector<Finding>>> &vFindingsPairs, double dValidationRatio) {
  if (dValidationRatio <= 0.0) {
    std::shuffle(vFindingsPairs.begin(), vFindingsPairs.end(), GetGenerator());
    return vFindingsPairs.size();
  }

  if (dValidationRatio >= 1.0) {
    std::shuffle(vFindingsPairs.begin(), vFindingsPairs.end(), GetGenerator());
    return 0;
  }

  // Partition into [positive|negative] cases
  const auto splitItr = std::partition(vFindingsPairs.begin(), vFindingsPairs.end(),
    [](const auto &stPair) -> bool {
      return std::find_if(stPair.second.begin(), stPair.second.end(),
        [](const Finding &stFinding) -> bool { 
          return stFinding.eLabel == Finding::True; 
        }) != stPair.second.end();
    });

  std::shuffle(vFindingsPairs.begin(), splitItr, GetGenerator());
  std::shuffle(splitItr, vFindingsPairs.end(), GetGenerator());

  const size_t posSize = (splitItr - vFindingsPairs.begin());
  const size_t negSize = (vFindingsPairs.end() - splitItr);

  std::cout << "Info: Clinically significant cases = " << posSize << ", negative cases = " << negSize << std::endl;

  // Calculate sizes
  const size_t validationPosSize = (size_t)(dValidationRatio * posSize);
  const size_t validationNegSize = (size_t)(dValidationRatio * negSize);

  const size_t trainingPosSize = posSize - validationPosSize;
  const size_t trainingNegSize = negSize - validationNegSize;

  // Iterator ranges
  const auto trainingPosBegin = vFindingsPairs.begin();
  const auto trainingPosEnd = trainingPosBegin + trainingPosSize;

  const auto trainingNegBegin = splitItr;
  const auto trainingNegEnd = trainingNegBegin + trainingNegSize;

  const auto validationPosBegin = trainingPosEnd;
  const auto validationPosEnd = validationPosBegin + validationPosSize;

  const auto validationNegBegin = trainingNegEnd;
  const auto validationNegEnd = validationNegBegin + validationNegSize;

  std::vector<std::pair<std::string, std::vector<Finding>>> vShuffledFindings(vFindingsPairs.size());

  auto outItr = vShuffledFindings.begin();

  outItr = std::move(trainingPosBegin, trainingPosEnd, outItr);
  outItr = std::move(trainingNegBegin, trainingNegEnd, outItr);

  const size_t trainingSize = outItr - vShuffledFindings.begin();

  outItr = std::move(validationPosBegin, validationPosEnd, outItr);
  outItr = std::move(validationNegBegin, validationNegEnd, outItr);

  // Now swap to return the shuffled findings
  vFindingsPairs.swap(vShuffledFindings);

  // Shuffle overall training/validation cases
  std::shuffle(vFindingsPairs.begin(), vFindingsPairs.begin() + trainingSize, GetGenerator());
  std::shuffle(vFindingsPairs.begin() + trainingSize, vFindingsPairs.end(), GetGenerator());

  return trainingSize;
}

template<typename PixelType>
typename itk::Image<PixelType, 3>::Pointer LoadT2WImage(const std::string &strPath) {
  // I store T2W cases in a hierarchy like
  // T2W/<patient id>/<study date>/<instance number>.dcm
  // ...
  // But the CSV file does not include a study date... so let's just find the first folder...

  std::vector<std::string> vFolders;
  FindDicomFolders(strPath.c_str(), "*", vFolders, true);

  if (vFolders.empty())
    return nullptr;

  return LoadDicomImage<PixelType, 3>(vFolders[0]);
}

template<typename PixelType>
std::vector<itk::IndexValueType> MakeMask(itk::Image<PixelType, 3> *p_clMask, const std::vector<Finding> &vFindings, unsigned int a_uiCounts[2]) {
  constexpr double dPositiveDistance = 5.0; // < 5 = clinical significance (+/-1)
  constexpr double dDontCareDistance = 8.0; // < 8 = -1 (we don't know lesion size!)
                                            // Everything else 0

  if (p_clMask == nullptr || vFindings.empty())
    return std::vector<itk::IndexValueType>();

  const itk::Size<3> clSize = p_clMask->GetBufferedRegion().GetSize();
  std::unordered_set<itk::IndexValueType> sSlices;

  for (itk::IndexValueType z = 0; itk::SizeValueType(z) < clSize[2]; ++z) {
    unsigned int a_uiTmpCounts[2] = { 0, 0 };

    for (itk::IndexValueType y = 0; itk::SizeValueType(y) < clSize[1]; ++y) {
      for (itk::IndexValueType x = 0; itk::SizeValueType(x) < clSize[0]; ++x) {
        const itk::Index<3> clIndex = {{ x, y, z }};

        Finding::PointType clVolumePosition;
        p_clMask->TransformIndexToPhysicalPoint(clIndex, clVolumePosition);

        auto minItr = std::min_element(vFindings.begin(), vFindings.end(),
          [&clVolumePosition](const Finding &a, const Finding &b) -> bool {
            return a.Distance2(clVolumePosition) < b.Distance2(clVolumePosition);
          });

        const double dDistanceToBiopsy = minItr->Distance(clVolumePosition);
        if (dDistanceToBiopsy < dPositiveDistance) {
          sSlices.insert(z);
          const short sLabel = minItr->eLabel == Finding::True ? 1 : 0;
          p_clMask->SetPixel(clIndex, sLabel);

          ++a_uiTmpCounts[sLabel];
        }
        else if (dDistanceToBiopsy < dDontCareDistance) {
          const short sLabel = minItr->eLabel == Finding::True ? -1 : 0;
          p_clMask->SetPixel(clIndex, sLabel);

          if (sLabel >= 0)
            ++a_uiTmpCounts[sLabel];
        }
        else {
          p_clMask->SetPixel(clIndex, 0);
          ++a_uiTmpCounts[0];
        }
      }
    }

    // Only count for slices used in training/validation
    if (sSlices.find(z) != sSlices.end()) {
      a_uiCounts[0] += a_uiTmpCounts[0];
      a_uiCounts[1] += a_uiTmpCounts[1];
    }
  }

  std::vector<itk::IndexValueType> vSlices(sSlices.begin(), sSlices.end());
  std::sort(vSlices.begin(), vSlices.end());

  return vSlices;
}
