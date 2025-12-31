import 'package:flutter/material.dart';
import 'dart:math';
import 'dart:async';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: '로또 마스터',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1A237E),
          secondary: const Color(0xFFFFD700),
          surface: Colors.grey[50]!,
        ),
        useMaterial3: true,
        fontFamily: 'Pretendard',
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
          backgroundColor: Colors.white,
          titleTextStyle: TextStyle(
            color: Color(0xFF1A237E),
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
          iconTheme: IconThemeData(color: Color(0xFF1A237E)),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 2,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),
      home: const RootScreen(),
    );
  }
}

class RootScreen extends StatefulWidget {
  const RootScreen({super.key});

  @override
  State<RootScreen> createState() => _RootScreenState();
}

class _RootScreenState extends State<RootScreen> {
  int _currentIndex = 0;
  List<Map<String, dynamic>> _savedNumbers = []; // 번호 + 날짜 + 회차 정보

  @override
  void initState() {
    super.initState();
    _loadNumbers();
  }

  Future<void> _loadNumbers() async {
    final prefs = await SharedPreferences.getInstance();
    final String? jsonString = prefs.getString('lotto_numbers');
    if (jsonString != null) {
      try {
        final List<dynamic> jsonList = jsonDecode(jsonString);
        setState(() {
          _savedNumbers = jsonList.map((e) {
            // 구버전 데이터 호환 (List<int>만 있는 경우)
            if (e is List) {
              return {
                'numbers': List<int>.from(e),
                'date': DateTime.now().toIso8601String(),
                'round': _calculateCurrentRound(),
                'type': 'random', // 기본값
              };
            }
            // 신버전 데이터 (Map)
            final map = Map<String, dynamic>.from(e);
            // 타입 정보가 없으면 기본값 설정
            if (!map.containsKey('type')) {
              map['type'] = 'random';
            }
            return map;
          }).toList();
        });
      } catch (e) {
        debugPrint('Error loading numbers: $e');
      }
    }
  }

  int _calculateCurrentRound() {
    // 기준: 2025년 12월 28일 (토) = 제1205회
    final referenceDate = DateTime(2025, 12, 28);
    const int referenceRound = 1205;
    
    final now = DateTime.now();
    final daysSinceReference = now.difference(referenceDate).inDays;
    final weeksSinceReference = daysSinceReference ~/ 7;
    
    int currentRound = referenceRound + weeksSinceReference;
    
    // 토요일 오후 8:35 이후면 다음 회차
    if (now.weekday == DateTime.saturday) {
      final drawTime = DateTime(now.year, now.month, now.day, 20, 35);
      if (now.isAfter(drawTime)) {
        currentRound += 1;
      }
    } else if (now.weekday == DateTime.sunday) {
      currentRound += 1;
    }
    
    return currentRound;
  }

  Future<void> _saveToPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final String jsonString = jsonEncode(_savedNumbers);
    await prefs.setString('lotto_numbers', jsonString);
  }

  void _addNumber(List<int> numbers, String type) {
    final now = DateTime.now();
    setState(() {
      _savedNumbers.insert(0, {
        'numbers': numbers,
        'date': now.toIso8601String(),
        'round': _calculateCurrentRound(),
        'type': type, // 'ai' or 'random'
      });
    });
    _saveToPrefs();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('번호가 저장 보관함에 추가되었습니다!'),
        duration: Duration(seconds: 1),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _addMultiple(List<List<int>> numbersList, String type) {
    final now = DateTime.now();
    final currentRound = _calculateCurrentRound();
    
    setState(() {
      // 역순으로 insert해서 순서 유지 (A조합이 맨 위에 오도록)
      for (var numbers in numbersList.reversed) {
        _savedNumbers.insert(0, {
          'numbers': numbers,
          'date': now.toIso8601String(),
          'round': currentRound,
          'type': type, // 'ai' or 'random'
        });
      }
    });
    _saveToPrefs();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('선택한 ${numbersList.length}개 번호가 저장되었습니다!'),
        duration: const Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _deleteNumber(int index) {
    setState(() {
      _savedNumbers.removeAt(index);
    });
    _saveToPrefs();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('번호가 삭제되었습니다.'),
        duration: Duration(seconds: 1),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _deleteMultiple(List<int> indices) {
    // 역순으로 정렬해서 삭제 (인덱스 변경 방지)
    final sortedIndices = indices.toList()..sort((a, b) => b.compareTo(a));
    setState(() {
      for (int idx in sortedIndices) {
        _savedNumbers.removeAt(idx);
      }
    });
    _saveToPrefs();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('선택한 ${indices.length}개 번호가 삭제되었습니다.'),
        duration: const Duration(seconds: 1),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _currentIndex == 0
          ? GeneratorScreen(
              onSave: (numbers, type) => _addNumber(numbers, type),
              onSaveMultiple: (numbersList, type) => _addMultiple(numbersList, type),
            )
          : HistoryScreen(
              savedNumbers: _savedNumbers,
              onDelete: _deleteNumber,
              onDeleteMultiple: _deleteMultiple,
            ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.auto_awesome_outlined),
            selectedIcon: Icon(Icons.auto_awesome),
            label: '번호 생성',
          ),
          NavigationDestination(
            icon: Icon(Icons.history_edu_outlined),
            selectedIcon: Icon(Icons.history_edu),
            label: '저장 보관함',
          ),
        ],
      ),
    );
  }
}

enum GenMode { random, analysis }

class GeneratorScreen extends StatefulWidget {
  final Function(List<int>, String) onSave;
  final Function(List<List<int>>, String) onSaveMultiple;

  const GeneratorScreen({
    super.key,
    required this.onSave,
    required this.onSaveMultiple,
  });

  @override
  State<GeneratorScreen> createState() => _GeneratorScreenState();
}

class _GeneratorScreenState extends State<GeneratorScreen> {
  List<List<int>> _currentNumbers = [];
  bool _isGenerating = false;
  GenMode _selectedMode = GenMode.random;
  Set<int> _selectedIndices = {}; // 선택된 번호들의 인덱스
  
  // 현재 진행 중인 회차를 자동으로 계산 (다음주 추첨 예정 회차)
  int get _totalRounds {
    // 기준: 2025년 12월 28일 (토) = 제1205회
    final referenceDate = DateTime(2025, 12, 28);
    const int referenceRound = 1205;
    
    final now = DateTime.now();
    final daysSinceReference = now.difference(referenceDate).inDays;
    final weeksSinceReference = daysSinceReference ~/ 7;
    
    int currentRound = referenceRound + weeksSinceReference;
    
    // 토요일 오후 8:35 이후면 다음 회차
    if (now.weekday == DateTime.saturday) {
      final drawTime = DateTime(now.year, now.month, now.day, 20, 35);
      if (now.isAfter(drawTime)) {
        currentRound += 1;
      }
    } else if (now.weekday == DateTime.sunday) {
      currentRound += 1;
    }
    
    return currentRound;
  }

  // 다음 추첨일 계산
  DateTime get _nextDrawDate {
    final now = DateTime.now();
    
    // 오늘이 토요일인 경우
    if (now.weekday == DateTime.saturday) {
      final drawTime = DateTime(now.year, now.month, now.day, 20, 35);
      if (now.isBefore(drawTime)) {
        return drawTime; // 오늘 추첨
      } else {
        return now.add(const Duration(days: 7)).copyWith(hour: 20, minute: 35); // 다음주 토요일
      }
    }
    
    // 일요일~금요일: 다가오는 토요일
    int daysUntilSaturday = (DateTime.saturday - now.weekday) % 7;
    if (daysUntilSaturday == 0) daysUntilSaturday = 7;
    
    final nextSaturday = now.add(Duration(days: daysUntilSaturday));
    return DateTime(nextSaturday.year, nextSaturday.month, nextSaturday.day, 20, 35);
  }

  // API 서버 주소 설정
  // ngrok 공개 주소 (어디서나 접속 가능!)
  static const String API_BASE_URL = 'https://expansional-hosea-drippily.ngrok-free.dev';
  
  // 다른 옵션들:
  // 에뮬레이터(로컬): http://10.0.2.2:5000
  // 같은 Wi-Fi: http://172.20.10.5:5000
  // AWS 배포 후: https://[AWS 도메인]
  
  // 2024년 12월 28일 기준 (제1152회) 실제 누적 당첨 횟수 통계 (정확한 데이터)
  // 출처: 동행복권 공식 데이터 기반
  final Map<int, int> _numberWeights = {
    1: 190, 2: 177, 3: 179, 4: 184, 5: 168,
    6: 182, 7: 183, 8: 167, 9: 145, 10: 176,
    11: 178, 12: 195, 13: 188, 14: 185, 15: 173,
    16: 180, 17: 191, 18: 185, 19: 171, 20: 186,
    21: 178, 22: 149, 23: 164, 24: 182, 25: 169,
    26: 182, 27: 193, 28: 166, 29: 156, 30: 169,
    31: 177, 32: 163, 33: 187, 34: 202, 35: 173,
    36: 175, 37: 184, 38: 182, 39: 185, 40: 183,
    41: 163, 42: 170, 43: 197, 44: 175, 45: 185
  };

  List<int> get _topFrequentNumbers {
    var sortedKeys = _numberWeights.keys.toList()
      ..sort((a, b) => _numberWeights[b]!.compareTo(_numberWeights[a]!));
    return sortedKeys.take(5).toList();
  }

  // 단순 랜덤
  List<int> _generateRandomSet() {
    final random = Random();
    final Set<int> numbers = {};
    while (numbers.length < 6) {
      numbers.add(random.nextInt(45) + 1);
    }
    return numbers.toList()..sort();
  }

  // [고급 AI 시뮬레이션 알고리즘]
  // 1. 가중치 기반 랜덤 추출 (룰렛 휠)
  // 2. 홀짝 비율 필터링 (너무 한쪽으로 쏠리면 재추첨)
  // 3. 연속 번호 패턴 고려
  List<int> _generateWeightedSet() {
    final random = Random();
    List<int> result = [];
    
    // 유효한 조합이 나올 때까지 반복 (최대 10번 시도)
    for (int i = 0; i < 10; i++) {
      result = _tryGenerateWeighted(random);
      
      // 홀짝 비율 검사 (짝수나 홀수가 6개 모두 나오는 극단적 상황 방지)
      int oddCount = result.where((n) => n % 2 != 0).length;
      if (oddCount >= 1 && oddCount <= 5) {
        break; // 적절한 비율이면 채택
      }
      // 아니면 다시 뽑기
    }
    
    return result..sort();
  }

  List<int> _tryGenerateWeighted(Random random) {
    final Set<int> selected = {};
    Map<int, int> currentWeights = Map.from(_numberWeights);

    // Top 5 번호 중 1~2개를 40% 확률로 우선 포함 (Hot Number 전략)
    if (random.nextDouble() < 0.4) {
       selected.add(_topFrequentNumbers[random.nextInt(5)]);
    }

    while (selected.length < 6) {
      int totalWeight = currentWeights.values.fold(0, (sum, weight) => sum + weight);
      int randomValue = random.nextInt(totalWeight);
      int currentSum = 0;
      int pickedNumber = -1;

      for (var entry in currentWeights.entries) {
        currentSum += entry.value;
        if (randomValue < currentSum) {
          pickedNumber = entry.key;
          break;
        }
      }

      if (pickedNumber != -1) {
        selected.add(pickedNumber);
        currentWeights.remove(pickedNumber);
      }
    }
    return selected.toList();
  }

  // AI 모델 API 호출 함수
  Future<List<int>?> _fetchAIRecommendation() async {
    try {
      final response = await http.get(
        Uri.parse('$API_BASE_URL/predict'),
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'success' && data['numbers'] != null) {
          List<int> numbers = List<int>.from(data['numbers']);
          return numbers;
        }
      }
      return null;
    } catch (e) {
      debugPrint('API 호출 실패: $e');
      return null;
    }
  }

  Future<void> _generate(int count) async {
    setState(() {
      _isGenerating = true;
      _currentNumbers = [];
      _selectedIndices.clear(); // 선택 초기화
    });

    List<List<int>> newSets = [];
    
    if (_selectedMode == GenMode.analysis) {
      // AI 모델 API만 사용 (로컬 알고리즘 폴백 없음)
      bool hasError = false;
      
      for (int i = 0; i < count; i++) {
        List<int>? aiNumbers = await _fetchAIRecommendation();
        
        if (aiNumbers != null && aiNumbers.length == 6) {
          newSets.add(aiNumbers);
        } else {
          hasError = true;
          break;
        }
      }
      
      setState(() {
        _isGenerating = false;
      });
      
      if (hasError) {
        // API 실패 시 에러 메시지 표시
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('AI 서버와 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.'),
              duration: Duration(seconds: 3),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }
    } else {
      // 랜덤 모드
      await Future.delayed(const Duration(milliseconds: 800));
      for (int i = 0; i < count; i++) {
        newSets.add(_generateRandomSet());
      }
      
      setState(() {
        _isGenerating = false;
      });
    }

    setState(() {
      _currentNumbers = newSets;
    });
  }

  @override
  Widget build(BuildContext context) {
    final isAnalysis = _selectedMode == GenMode.analysis;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Lotto Master'),
        actions: [
           IconButton(
            icon: const Icon(Icons.help_outline),
            onPressed: () {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('추첨 방식 안내', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  content: const Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('🍀 ', style: TextStyle(fontSize: 20)),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('운에 맡기기', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
                                SizedBox(height: 4),
                                Text('완전 무작위로 번호를 생성합니다.\n복권 구매 시 자동과 동일한 방식입니다.', style: TextStyle(fontSize: 14, height: 1.4)),
                              ],
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: 16),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('🧠 ', style: TextStyle(fontSize: 20)),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('빅데이터 분석', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
                                SizedBox(height: 4),
                                Text('제1회부터 최신 회차까지 모든 당첨번호를 학습한 인공지능이 과거 패턴을 분석하여 추천 번호를 생성합니다.\n\n※ 딥러닝(LSTM) 기술 적용', style: TextStyle(fontSize: 14, height: 1.4)),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('확인', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold)),
                    )
                  ],
                ),
              );
            },
          )
        ],
      ),
      body: Column(
        children: [
          const SizedBox(height: 16),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: SegmentedButton<GenMode>(
              segments: const [
                ButtonSegment<GenMode>(
                  value: GenMode.random,
                  label: Text('운에 맡기기'),
                  icon: Icon(Icons.casino),
                ),
                ButtonSegment<GenMode>(
                  value: GenMode.analysis,
                  label: Text('빅데이터 분석'),
                  icon: Icon(Icons.analytics),
                ),
              ],
              selected: {_selectedMode},
              onSelectionChanged: (Set<GenMode> newSelection) {
                setState(() {
                  _selectedMode = newSelection.first;
                  _currentNumbers = [];
                });
              },
              style: ButtonStyle(
                backgroundColor: MaterialStateProperty.resolveWith<Color>((states) {
                  if (states.contains(MaterialState.selected)) {
                    return isAnalysis ? const Color(0xFF1A237E).withOpacity(0.1) : Colors.green.withOpacity(0.1);
                  }
                  return Colors.transparent;
                }),
                foregroundColor: MaterialStateProperty.resolveWith<Color>((states) {
                   if (states.contains(MaterialState.selected)) {
                    return isAnalysis ? const Color(0xFF1A237E) : Colors.green;
                  }
                  return Colors.grey;
                }),
              ),
            ),
          ),
          
          const SizedBox(height: 12),

          // 다음 추첨일 안내
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    const Color(0xFFFFD700).withOpacity(0.1),
                    const Color(0xFFFFA500).withOpacity(0.1),
                  ],
                ),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: const Color(0xFFFFD700).withOpacity(0.3),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.celebration,
                    size: 18,
                    color: Color(0xFFFF6F00),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '다음 추첨: ${_nextDrawDate.month}/${_nextDrawDate.day} (토) 오후 8:35',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFFFF6F00),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFF6F00).withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '제${_totalRounds}회',
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFFFF6F00),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          const SizedBox(height: 12),

          Expanded(
            child: _isGenerating
                ? LottoDrawAnimation(isAnalysis: isAnalysis)
                : _currentNumbers.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              isAnalysis ? Icons.psychology_outlined : Icons.casino_outlined,
                              size: 100,
                              color: Colors.grey[300],
                            ),
                            const SizedBox(height: 24),
                            Text(
                              isAnalysis
                                  ? '인공지능이 과거 데이터를 분석하여\n추천번호를 생성해드립니다'
                                  : '행운의 번호를 뽑아보세요!\n아래 버튼을 눌러 시작하세요',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.grey[600], 
                                fontSize: 16,
                                height: 1.5,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                            const SizedBox(height: 12),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                              decoration: BoxDecoration(
                                color: isAnalysis ? const Color(0xFF1A237E).withOpacity(0.1) : Colors.green.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Text(
                                isAnalysis ? '👇 아래 버튼을 눌러주세요' : '👇 아래 버튼을 눌러주세요',
                                style: TextStyle(
                                  color: isAnalysis ? const Color(0xFF1A237E) : Colors.green,
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ],
                        ),
                      )
                    : Column(
                        children: [
                          // 전체 선택 버튼
                          if (_currentNumbers.isNotEmpty)
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    '${_selectedIndices.length}개 선택됨',
                                    style: TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.w600,
                                      color: Colors.grey[700],
                                    ),
                                  ),
                                  TextButton.icon(
                                    onPressed: () {
                                      setState(() {
                                        if (_selectedIndices.length == _currentNumbers.length) {
                                          // 전체 해제
                                          _selectedIndices.clear();
                                        } else {
                                          // 전체 선택
                                          _selectedIndices = Set.from(
                                            List.generate(_currentNumbers.length, (i) => i)
                                          );
                                        }
                                      });
                                    },
                                    icon: Icon(
                                      _selectedIndices.length == _currentNumbers.length 
                                        ? Icons.check_box 
                                        : Icons.check_box_outline_blank,
                                      size: 20,
                                    ),
                                    label: Text(
                                      _selectedIndices.length == _currentNumbers.length 
                                        ? '전체 해제' 
                                        : '전체 선택',
                                      style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                                    ),
                                    style: TextButton.styleFrom(
                                      foregroundColor: const Color(0xFF1A237E),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          Expanded(
                            child: Scrollbar(
                              thumbVisibility: true,
                              thickness: 6,
                              radius: const Radius.circular(10),
                              child: ListView.builder(
                                padding: const EdgeInsets.fromLTRB(16, 0, 16, 100),
                                itemCount: isAnalysis ? _currentNumbers.length + 1 : _currentNumbers.length,
                                itemBuilder: (context, index) {
                            // 빅데이터 분석 모드일 때 첫 번째 항목은 AI 안내 박스
                            if (isAnalysis && index == 0) {
                              return Container(
                                margin: const EdgeInsets.only(bottom: 12),
                                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [
                                      const Color(0xFF1A237E),
                                      const Color(0xFF283593),
                                    ],
                                  ),
                                  borderRadius: BorderRadius.circular(12),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.indigo.withOpacity(0.3),
                                      blurRadius: 8,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: Row(
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        color: Colors.white.withOpacity(0.2),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: const Icon(Icons.psychology, color: Colors.amber, size: 24),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          const Text(
                                            '인공지능 분석 번호',
                                            style: TextStyle(
                                              color: Colors.white,
                                              fontWeight: FontWeight.bold,
                                              fontSize: 15,
                                              letterSpacing: -0.3,
                                            ),
                                          ),
                                          const SizedBox(height: 4),
                                          Text(
                                            '제1회~제$_totalRounds회 전체 데이터 학습 완료',
                                            style: TextStyle(
                                              color: Colors.amberAccent,
                                              fontSize: 11,
                                              fontWeight: FontWeight.w600,
                                            ),
                                          ),
                                          const SizedBox(height: 2),
                                          Text(
                                            '과거 패턴 분석 · 딥러닝 모델 AI 추천',
                                            style: TextStyle(
                                              color: Colors.white.withOpacity(0.85),
                                              fontSize: 11,
                                              height: 1.3,
                                            ),
                                          ),
                                        ],
                                      ),
                                    )
                                  ],
                                ),
                              );
                            }
                            // 실제 로또 번호 행
                            final lottoIndex = isAnalysis ? index - 1 : index;
                            return _buildLottoRow(
                              _currentNumbers[lottoIndex],
                              lottoIndex + 1,
                              isAnalysis,
                            );
                                },
                              ),
                            ),
                          ),
                        ],
                      ),
          ),
        ],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: Padding(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
        child: _currentNumbers.isNotEmpty && _selectedIndices.isNotEmpty
            ? Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // 선택한 것만 저장 버튼
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton.icon(
                      onPressed: () {
                        // 선택한 번호들을 리스트로 모아서 한 번에 저장 (로그 한 번만!)
                        final selectedNumbers = _selectedIndices
                            .map((idx) => _currentNumbers[idx])
                            .toList();
                        
                        final type = _selectedMode == GenMode.analysis ? 'ai' : 'random';
                        widget.onSaveMultiple(selectedNumbers, type);
                        
                        setState(() {
                          _selectedIndices.clear();
                        });
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                        foregroundColor: Colors.white,
                        elevation: 3,
                      ),
                      icon: const Icon(Icons.bookmark_add, size: 22),
                      label: Text(
                        '선택한 ${_selectedIndices.length}개 저장',
                        style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                ],
              )
            : isAnalysis
            ? 
            // 빅데이터 분석 모드: 5개 생성
            SizedBox(
                width: double.infinity,
                height: 54,
                child: ElevatedButton.icon(
                  onPressed: _isGenerating ? null : () => _generate(5),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A237E),
                    foregroundColor: Colors.white,
                    elevation: 3,
                  ),
                  icon: const Icon(Icons.auto_awesome, size: 24),
                  label: const Text(
                    'AI 추천번호 5개 생성',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: -0.5),
                  ),
                ),
              )
            : 
            // 운에 맡기기 모드: 1개 / 5개 선택 가능
            Row(
                children: [
                  Expanded(
                    child: SizedBox(
                      height: 54,
                      child: ElevatedButton.icon(
                        onPressed: _isGenerating ? null : () => _generate(1),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: Colors.green,
                          side: const BorderSide(color: Colors.green, width: 1.5),
                          elevation: 2,
                        ),
                        icon: const Icon(Icons.casino, size: 22),
                        label: const Text('번호 1개\n생성', 
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, height: 1.2)),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: SizedBox(
                      height: 54,
                      child: ElevatedButton.icon(
                        onPressed: _isGenerating ? null : () => _generate(5),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
                          elevation: 3,
                        ),
                        icon: const Icon(Icons.shuffle, size: 24),
                        label: const Text(
                          '번호 5개 한번에 생성',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: -0.5),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildLottoRow(List<int> numbers, int index, bool isAnalysis) {
    final now = DateTime.now();
    final currentRound = _totalRounds; // getter로 자동 계산됨
    final displayDate = '${now.year}.${now.month.toString().padLeft(2, '0')}.${now.day.toString().padLeft(2, '0')}';
    final drawDate = _nextDrawDate; // 추첨일
    final drawDateStr = '${drawDate.month}/${drawDate.day} (토)';
    
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isAnalysis ? const Color(0xFF1A237E).withOpacity(0.1) : Colors.grey.withOpacity(0.2),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.08),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    // 체크박스 추가
                    Checkbox(
                      value: _selectedIndices.contains(index - 1),
                      onChanged: (bool? value) {
                        setState(() {
                          if (value == true) {
                            _selectedIndices.add(index - 1);
                          } else {
                            _selectedIndices.remove(index - 1);
                          }
                        });
                      },
                      activeColor: const Color(0xFF1A237E),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                              decoration: BoxDecoration(
                                color: Colors.grey[100],
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: Text(
                                '${String.fromCharCode(65 + (index - 1))}조합',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 13,
                                  color: Colors.grey[700],
                                ),
                              ),
                            ),
                            if (isAnalysis) ...[
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 4),
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [
                                      const Color(0xFF1A237E),
                                      const Color(0xFF283593),
                                    ],
                                  ),
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: const Row(
                                  children: [
                                    Icon(Icons.auto_awesome, size: 11, color: Colors.amber),
                                    SizedBox(width: 3),
                                    Text(
                                      'AI 추천',
                                      style: TextStyle(
                                        fontSize: 11,
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ]
                          ],
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(
                                color: const Color(0xFF1A237E).withOpacity(0.1),
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Text(
                                '제${currentRound}회',
                                style: const TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                  color: Color(0xFF1A237E),
                                ),
                              ),
                            ),
                            const SizedBox(width: 6),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(
                                color: Colors.orange.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Row(
                                children: [
                                  const Icon(
                                    Icons.celebration,
                                    size: 10,
                                    color: Colors.orange,
                                  ),
                                  const SizedBox(width: 3),
                                  Text(
                                    '추첨 $drawDateStr',
                                    style: const TextStyle(
                                      fontSize: 10,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.orange,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
                TextButton.icon(
                  icon: const Icon(Icons.bookmark_border_rounded, size: 18),
                  label: const Text(
                    '저장',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  onPressed: () {
                    final type = _selectedMode == GenMode.analysis ? 'ai' : 'random';
                    widget.onSave(numbers, type);
                  },
                  style: TextButton.styleFrom(
                    foregroundColor: const Color(0xFF1A237E),
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: numbers.map((n) => LottoBall(number: n)).toList(),
            ),
          ],
        ),
      ),
    );
  }
}

class HistoryScreen extends StatefulWidget {
  final List<Map<String, dynamic>> savedNumbers;
  final Function(int) onDelete;
  final Function(List<int>) onDeleteMultiple;

  const HistoryScreen({
    super.key,
    required this.savedNumbers,
    required this.onDelete,
    required this.onDeleteMultiple,
  });

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  Set<int> _selectedIndices = {};
  Map<String, bool> _expandedDates = {}; // 날짜별 접기/펼치기 상태
  Map<String, bool> _expandedTypes = {}; // 타입별 접기/펼치기 상태 (날짜+타입 키)
  
  DateTime _getDrawDateForRound(int round) {
    // 기준: 제1205회 = 2025년 12월 28일 (토)
    final referenceRound = 1205;
    final referenceDate = DateTime(2025, 12, 28);
    
    final weeksDiff = round - referenceRound;
    final drawDate = referenceDate.add(Duration(days: weeksDiff * 7));
    
    return DateTime(drawDate.year, drawDate.month, drawDate.day, 20, 35);
  }
  
  // 날짜별로 그룹화, 각 날짜 안에서 타입별로 다시 그룹화
  Map<String, Map<String, List<Map<String, dynamic>>>> _groupByDateAndType() {
    final grouped = <String, Map<String, List<Map<String, dynamic>>>>{};
    
    for (var item in widget.savedNumbers) {
      final dateStr = item['date'] as String?;
      final type = item['type'] as String? ?? 'random'; // 기본값은 random
      
      if (dateStr != null) {
        try {
          final date = DateTime.parse(dateStr);
          final dateKey = '${date.year}.${date.month.toString().padLeft(2, '0')}.${date.day.toString().padLeft(2, '0')}';
          
          if (!grouped.containsKey(dateKey)) {
            grouped[dateKey] = {'ai': [], 'random': []};
            _expandedDates[dateKey] = true; // 기본적으로 펼쳐짐
          }
          
          if (!grouped[dateKey]!.containsKey(type)) {
            grouped[dateKey]![type] = [];
          }
          
          final typeKey = '$dateKey-$type';
          if (!_expandedTypes.containsKey(typeKey)) {
            _expandedTypes[typeKey] = true; // 기본적으로 펼쳐짐
          }
          
          grouped[dateKey]![type]!.add(item);
        } catch (e) {
          // 날짜 파싱 실패 시 무시
        }
      }
    }
    
    // 날짜 순으로 정렬 (최신순)
    final sortedKeys = grouped.keys.toList()
      ..sort((a, b) => b.compareTo(a));
    
    final sortedMap = <String, Map<String, List<Map<String, dynamic>>>>{};
    for (var key in sortedKeys) {
      sortedMap[key] = grouped[key]!;
    }
    
    return sortedMap;
  }
  
  Widget _buildGroupedListView() {
    final grouped = _groupByDateAndType();
    final dateKeys = grouped.keys.toList();
    
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: dateKeys.length,
      itemBuilder: (context, dateIndex) {
        final dateKey = dateKeys[dateIndex];
        final typeGroups = grouped[dateKey]!;
        final isExpanded = _expandedDates[dateKey] ?? true;
        
        // 전체 개수 계산
        final totalCount = (typeGroups['ai']?.length ?? 0) + (typeGroups['random']?.length ?? 0);
        
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          elevation: 1,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(color: Colors.grey.withOpacity(0.1), width: 1),
          ),
          child: ExpansionTile(
            initiallyExpanded: isExpanded,
            onExpansionChanged: (expanded) {
              setState(() {
                _expandedDates[dateKey] = expanded;
              });
            },
            leading: Icon(
              isExpanded ? Icons.folder_open : Icons.folder,
              color: const Color(0xFF1A237E),
              size: 24,
            ),
            title: Row(
              children: [
                Text(
                  dateKey,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF1A237E),
                  ),
                ),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1A237E).withOpacity(0.08),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '${totalCount}개',
                    style: const TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF1A237E),
                    ),
                  ),
                ),
              ],
            ),
            tilePadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            childrenPadding: EdgeInsets.zero,
            children: [
              // AI 추천 섹션
              if ((typeGroups['ai']?.length ?? 0) > 0)
                _buildTypeSection(dateKey, 'ai', typeGroups['ai']!, '빅데이터 분석'),
              // 운에 맞기기 섹션
              if ((typeGroups['random']?.length ?? 0) > 0)
                _buildTypeSection(dateKey, 'random', typeGroups['random']!, '운에 맞기기'),
            ],
          ),
        );
      },
    );
  }
  
  Widget _buildTypeSection(String dateKey, String type, List<Map<String, dynamic>> items, String typeLabel) {
    final typeKey = '$dateKey-$type';
    final isExpanded = _expandedTypes[typeKey] ?? true;
    
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.grey.withOpacity(0.03),
        borderRadius: BorderRadius.circular(8),
      ),
      child: ExpansionTile(
        initiallyExpanded: isExpanded,
        onExpansionChanged: (expanded) {
          setState(() {
            _expandedTypes[typeKey] = expanded;
          });
        },
        leading: Icon(
          isExpanded ? Icons.expand_less : Icons.expand_more,
          color: type == 'ai' ? Colors.orange[700] : Colors.blue[700],
          size: 20,
        ),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
              decoration: BoxDecoration(
                color: type == 'ai' 
                  ? Colors.orange.withOpacity(0.12)
                  : Colors.blue.withOpacity(0.12),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                typeLabel,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: type == 'ai' ? Colors.orange[800] : Colors.blue[800],
                ),
              ),
            ),
            const SizedBox(width: 6),
            Text(
              '${items.length}개',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w500,
                color: Colors.grey[600],
              ),
            ),
          ],
        ),
        tilePadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 0),
        childrenPadding: const EdgeInsets.only(bottom: 8),
      children: items.asMap().entries.map((entry) {
              final itemIndex = entry.key;
              final item = entry.value;
              final numbers = List<int>.from(item['numbers']);
              final round = item['round'] as int?;
              
              // 전체 리스트에서의 실제 인덱스 찾기
              final globalIndex = widget.savedNumbers.indexOf(item);
              final isSelected = _selectedIndices.contains(globalIndex);

              return Dismissible(
                key: Key(numbers.toString() + globalIndex.toString()),
                direction: DismissDirection.endToStart,
                onDismissed: (_) => widget.onDelete(globalIndex),
                background: Container(
                  alignment: Alignment.centerRight,
                  padding: const EdgeInsets.only(right: 20),
                  color: Colors.red,
                  child: const Icon(Icons.delete, color: Colors.white),
                ),
                child: Container(
                  margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: isSelected 
                      ? const Color(0xFF1A237E).withOpacity(0.08) 
                      : Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: isSelected 
                        ? const Color(0xFF1A237E).withOpacity(0.3)
                        : Colors.grey.withOpacity(0.15),
                      width: 1,
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: Row(
                                children: [
                                  Checkbox(
                                    value: isSelected,
                                    onChanged: (bool? value) {
                                      setState(() {
                                        if (value == true) {
                                          _selectedIndices.add(globalIndex);
                                        } else {
                                          _selectedIndices.remove(globalIndex);
                                        }
                                      });
                                    },
                                    activeColor: const Color(0xFF1A237E),
                                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                    visualDensity: VisualDensity.compact,
                                  ),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Row(
                                          children: [
                                            if (round != null) ...[
                                              Container(
                                                padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                                                decoration: BoxDecoration(
                                                  color: const Color(0xFF1A237E).withOpacity(0.1),
                                                  borderRadius: BorderRadius.circular(4),
                                                ),
                                                child: Text(
                                                  '제${round}회',
                                                  style: const TextStyle(
                                                    fontSize: 10,
                                                    fontWeight: FontWeight.w600,
                                                    color: Color(0xFF1A237E),
                                                  ),
                                                ),
                                              ),
                                            ],
                                          ],
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            IconButton(
                              icon: const Icon(Icons.delete_outline, color: Colors.grey, size: 18),
                              onPressed: () {
                                showDialog(
                                  context: context,
                                  builder: (context) => AlertDialog(
                                    title: const Text('삭제 확인'),
                                    content: const Text('이 번호를 영구적으로 삭제하시겠습니까?'),
                                    actions: [
                                      TextButton(
                                        onPressed: () => Navigator.pop(context),
                                        child: const Text('취소'),
                                      ),
                                      TextButton(
                                        onPressed: () {
                                          Navigator.pop(context);
                                          widget.onDelete(globalIndex);
                                        },
                                        child: const Text('삭제', style: TextStyle(color: Colors.red)),
                                      ),
                                    ],
                                  ),
                                );
                              },
                              padding: EdgeInsets.zero,
                              constraints: const BoxConstraints(),
                              tooltip: '삭제',
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: numbers.map((n) => LottoBall(number: n)).toList(),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            }).toList(),
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          _selectedIndices.isEmpty ? '저장 보관함' : '${_selectedIndices.length}개 선택됨',
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        actions: [
          // 전체 선택/해제 버튼
          if (widget.savedNumbers.isNotEmpty)
            IconButton(
              onPressed: () {
                setState(() {
                  if (_selectedIndices.length == widget.savedNumbers.length) {
                    _selectedIndices.clear();
                  } else {
                    _selectedIndices = Set.from(
                      List.generate(widget.savedNumbers.length, (i) => i)
                    );
                  }
                });
              },
              icon: Icon(
                _selectedIndices.length == widget.savedNumbers.length 
                  ? Icons.check_box 
                  : Icons.check_box_outline_blank,
              ),
              tooltip: _selectedIndices.length == widget.savedNumbers.length 
                  ? '전체 해제' 
                  : '전체 선택',
            ),
          if (_selectedIndices.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: () {
                showDialog(
                  context: context,
                  builder: (context) => AlertDialog(
                    title: const Text('일괄 삭제'),
                    content: Text('선택한 ${_selectedIndices.length}개 번호를 삭제하시겠습니까?'),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.pop(context),
                        child: const Text('취소'),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.pop(context);
                          widget.onDeleteMultiple(_selectedIndices.toList());
                          setState(() {
                            _selectedIndices.clear();
                          });
                        },
                        child: const Text('삭제', style: TextStyle(color: Colors.red)),
                      ),
                    ],
                  ),
                );
              },
              tooltip: '선택 삭제',
            ),
        ],
      ),
      body: widget.savedNumbers.isEmpty
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.folder_open_outlined, size: 64, color: Colors.grey[300]),
                  const SizedBox(height: 16),
                  const Text(
                    '저장된 번호가 없습니다.',
                    style: TextStyle(fontSize: 16, color: Colors.grey),
                  ),
                ],
              ),
            )
          : _buildGroupedListView(),
    );
  }
}

class LottoBall extends StatelessWidget {
  final int number;

  const LottoBall({super.key, required this.number});

  Color _getBallColor(int number) {
    if (number <= 10) return const Color(0xFFFBC400);
    if (number <= 20) return const Color(0xFF69C8F2);
    if (number <= 30) return const Color(0xFFFF7272);
    if (number <= 40) return const Color(0xFFAAAAAA);
    return const Color(0xFFB0D840);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 38,
      height: 38,
      decoration: BoxDecoration(
        color: _getBallColor(number),
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.12),
            blurRadius: 3,
            offset: const Offset(1, 2),
          ),
        ],
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            _getBallColor(number).withOpacity(0.8),
            _getBallColor(number),
          ],
        ),
      ),
      child: Center(
        child: Text(
          '$number',
          style: TextStyle(
            color: number <= 10 ? Colors.black : Colors.white,
            fontWeight: FontWeight.w900,
            fontSize: 15,
            shadows: [
              if (number > 10)
                const Shadow(
                  color: Colors.black26,
                  offset: Offset(1, 1),
                  blurRadius: 2,
                )
            ],
          ),
        ),
      ),
    );
  }
}

// 로또 추첨 애니메이션 위젯 (물리 엔진 적용)
class LottoDrawAnimation extends StatefulWidget {
  final bool isAnalysis;

  const LottoDrawAnimation({super.key, required this.isAnalysis});

  @override
  State<LottoDrawAnimation> createState() => _LottoDrawAnimationState();
}

class _LottoDrawAnimationState extends State<LottoDrawAnimation> {
  late Timer _timer;
  late List<_PhysicsBall> _balls;
  final double containerRadius = 130.0; // 컨테이너 반지름
  final double ballRadius = 18.0; // 공 반지름

  @override
  void initState() {
    super.initState();
    
    // 15개의 물리 공 생성 (적당한 개수)
    final random = Random();
    _balls = List.generate(15, (index) {
      // 랜덤 위치 (원 안쪽)
      double angle = random.nextDouble() * 2 * pi;
      double distance = random.nextDouble() * (containerRadius - ballRadius - 10);
      
      return _PhysicsBall(
        number: random.nextInt(45) + 1,
        x: 140 + cos(angle) * distance,
        y: 140 + sin(angle) * distance,
        vx: (random.nextDouble() - 0.5) * 6,
        vy: (random.nextDouble() - 0.5) * 6,
      );
    });
    
    // 물리 시뮬레이션 타이머 (60 FPS)
    _timer = Timer.periodic(const Duration(milliseconds: 16), (timer) {
      setState(() {
        _updatePhysics();
      });
    });
  }

  void _updatePhysics() {
    const double gravity = 0.3; // 중력 (아래로)
    const double friction = 0.993; // 공기 저항 (거의 없음)
    const double bounce = 0.9; // 탄성 (매우 튕김!)
    const double windStrength = 1.5; // 폭풍 같은 바람!
    const double turbulence = 0.8; // 강한 난기류!
    final random = Random();
    
    for (var ball in _balls) {
      // 중력 적용 (아래로)
      ball.vy += gravity;
      
      // 아래에서 위로 분수처럼 솟구치는 강력한 바람!
      double ballYFromCenter = ball.y - 140;
      double ballXFromCenter = ball.x - 140;
      
      // 바람은 오직 아래쪽에서만 강하게! (분수 효과)
      // y > 140이면 아래쪽, y < 140이면 위쪽
      if (ballYFromCenter > 20) { // 아래쪽 절반에만 바람 적용
        // 밑으로 갈수록 바람이 훨씬 강함 (지수적으로 증가!)
        double bottomDistance = ballYFromCenter; // 중심선 아래 거리
        double windPower = (bottomDistance / 140); // 0 ~ 1
        windPower = windPower * windPower; // 제곱으로 더 강하게!
        
        // 강력한 상승 바람 (분수처럼!)
        double upwardForce = windStrength * windPower * 2.0;
        ball.vy -= upwardForce;
        
        // 중심 쪽으로 약간 모았다가 위로 쏘는 효과 (분수의 물줄기처럼)
        if (ballYFromCenter > 60) { // 맨 아래쪽에서만
          ball.vx -= ballXFromCenter * 0.03; // 중심으로 모음
        }
        
        // 난기류는 아래쪽에서만 (바람의 불규칙성)
        ball.vx += (random.nextDouble() - 0.5) * turbulence * windPower;
        ball.vy += (random.nextDouble() - 0.5) * turbulence * windPower * 0.5;
      } else {
        // 위쪽에서는 바람이 거의 없음 (자유 낙하)
        // 약간의 회오리 효과만
        double angle = atan2(ballYFromCenter, ballXFromCenter);
        ball.vx += cos(angle + pi / 2) * 0.1;
        ball.vy += sin(angle + pi / 2) * 0.1;
        
        // 중심에서 살짝 밀어내기 (공들이 퍼지게)
        ball.vx += ballXFromCenter * 0.01;
        ball.vy += ballYFromCenter * 0.005;
      }
      
      // 속도 적용
      ball.x += ball.vx;
      ball.y += ball.vy;
      
      // 원형 벽 충돌 감지
      double dx = ball.x - 140;
      double dy = ball.y - 140;
      double distance = sqrt(dx * dx + dy * dy);
      
      if (distance + ballRadius > containerRadius) {
        // 벽과 충돌 시 반사
        double angle = atan2(dy, dx);
        ball.x = 140 + cos(angle) * (containerRadius - ballRadius);
        ball.y = 140 + sin(angle) * (containerRadius - ballRadius);
        
        // 속도 반사 (탄성 충돌)
        double normalX = dx / distance;
        double normalY = dy / distance;
        double dotProduct = ball.vx * normalX + ball.vy * normalY;
        
        ball.vx = (ball.vx - 2 * dotProduct * normalX) * bounce;
        ball.vy = (ball.vy - 2 * dotProduct * normalY) * bounce;
        
        // 벽에 부딪힐 때 강한 랜덤 힘 추가 (폭발적으로!)
        ball.vx += (random.nextDouble() - 0.5) * 1.5;
        ball.vy += (random.nextDouble() - 0.5) * 1.5;
      }
      
      // 속도 제한 (너무 빨라지지 않게)
      double speed = sqrt(ball.vx * ball.vx + ball.vy * ball.vy);
      if (speed > 8) {
        ball.vx = ball.vx / speed * 8;
        ball.vy = ball.vy / speed * 8;
      }
      
      // 최소한의 공기 저항만 적용
      ball.vx *= friction;
      ball.vy *= friction;
    }
    
    // 공끼리 충돌 감지 (강력한 반발력!)
    for (int i = 0; i < _balls.length; i++) {
      for (int j = i + 1; j < _balls.length; j++) {
        _PhysicsBall ball1 = _balls[i];
        _PhysicsBall ball2 = _balls[j];
        
        double dx = ball2.x - ball1.x;
        double dy = ball2.y - ball1.y;
        double distance = sqrt(dx * dx + dy * dy);
        
        if (distance < ballRadius * 2 && distance > 0) {
          // 충돌 시 강력하게 튕겨내기
          double angle = atan2(dy, dx);
          double overlap = ballRadius * 2 - distance;
          
          // 겹친 만큼 밀어내기
          double separateX = cos(angle) * overlap * 0.6;
          double separateY = sin(angle) * overlap * 0.6;
          
          ball1.x -= separateX;
          ball1.y -= separateY;
          ball2.x += separateX;
          ball2.y += separateY;
          
          // 속도 교환 (탄성 충돌) + 추가 반발력
          double normalX = dx / distance;
          double normalY = dy / distance;
          
          double relativeVx = ball2.vx - ball1.vx;
          double relativeVy = ball2.vy - ball1.vy;
          double dotProduct = relativeVx * normalX + relativeVy * normalY;
          
          double impulse = dotProduct * 1.2; // 반발력 증가!
          
          ball1.vx += impulse * normalX;
          ball1.vy += impulse * normalY;
          ball2.vx -= impulse * normalX;
          ball2.vy -= impulse * normalY;
          
          // 충돌 시 랜덤 힘 추가 (폭발 효과)
          ball1.vx += (random.nextDouble() - 0.5) * 0.8;
          ball1.vy += (random.nextDouble() - 0.5) * 0.8;
          ball2.vx += (random.nextDouble() - 0.5) * 0.8;
          ball2.vy += (random.nextDouble() - 0.5) * 0.8;
        }
      }
    }
  }

  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // 구형 추첨기
          Container(
            width: 280,
            height: 280,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                center: Alignment.topLeft,
                radius: 1.2,
                colors: [
                  Colors.white.withOpacity(0.6),
                  Colors.white.withOpacity(0.2),
                  Colors.grey.withOpacity(0.3),
                ],
              ),
              border: Border.all(
                color: Colors.white.withOpacity(0.6),
                width: 5,
              ),
              boxShadow: [
                BoxShadow(
                  color: (widget.isAnalysis
                          ? const Color(0xFF1A237E)
                          : Colors.green)
                      .withOpacity(0.3),
                  blurRadius: 30,
                  offset: const Offset(0, 15),
                ),
                BoxShadow(
                  color: Colors.black.withOpacity(0.15),
                  blurRadius: 20,
                  offset: const Offset(0, 10),
                ),
              ],
            ),
            child: ClipOval(
              child: Container(
                decoration: BoxDecoration(
                  gradient: RadialGradient(
                    colors: [
                      widget.isAnalysis
                          ? const Color(0xFF1A237E).withOpacity(0.03)
                          : Colors.green.withOpacity(0.03),
                      Colors.transparent,
                    ],
                  ),
                ),
                child: Stack(
                  children: _balls.map((ball) {
                    return Positioned(
                      left: ball.x - ballRadius,
                      top: ball.y - ballRadius,
                      child: _AnimatedLottoBall(
                        number: ball.number,
                        size: ballRadius * 2,
                      ),
                    );
                  }).toList(),
                ),
              ),
            ),
          ),
          
          const SizedBox(height: 40),
          
          // 로딩 표시
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            decoration: BoxDecoration(
              color: widget.isAnalysis
                  ? const Color(0xFF1A237E)
                  : Colors.green,
              borderRadius: BorderRadius.circular(30),
              boxShadow: [
                BoxShadow(
                  color: (widget.isAnalysis
                          ? const Color(0xFF1A237E)
                          : Colors.green)
                      .withOpacity(0.3),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2.5,
                    valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  widget.isAnalysis
                      ? 'AI가 번호를 분석하고 있습니다...'
                      : '행운의 번호를 뽑고 있습니다...',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 15,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Text(
            widget.isAnalysis
                ? '딥러닝 모델이 과거 패턴을 분석 중입니다'
                : '잠시만 기다려주세요',
            style: TextStyle(
              color: Colors.grey[600],
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}

// 물리 공 클래스
class _PhysicsBall {
  final int number;
  double x;
  double y;
  double vx; // x 방향 속도
  double vy; // y 방향 속도

  _PhysicsBall({
    required this.number,
    required this.x,
    required this.y,
    required this.vx,
    required this.vy,
  });
}

// 애니메이션용 로또 공
class _AnimatedLottoBall extends StatelessWidget {
  final int number;
  final double size;

  const _AnimatedLottoBall({
    required this.number,
    this.size = 45,
  });

  Color _getBallColor(int number) {
    if (number <= 10) return const Color(0xFFFBC400);
    if (number <= 20) return const Color(0xFF69C8F2);
    if (number <= 30) return const Color(0xFFFF7272);
    if (number <= 40) return const Color(0xFFAAAAAA);
    return const Color(0xFFB0D840);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: _getBallColor(number),
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 6,
            offset: const Offset(2, 3),
          ),
          BoxShadow(
            color: Colors.white.withOpacity(0.3),
            blurRadius: 3,
            offset: const Offset(-1, -1),
          ),
        ],
        gradient: RadialGradient(
          center: Alignment.topLeft,
          radius: 0.8,
          colors: [
            Colors.white.withOpacity(0.4),
            _getBallColor(number).withOpacity(0.9),
            _getBallColor(number),
          ],
          stops: const [0.0, 0.4, 1.0],
        ),
      ),
      child: Center(
        child: Text(
          '$number',
          style: TextStyle(
            color: number <= 10 ? Colors.black : Colors.white,
            fontWeight: FontWeight.w900,
            fontSize: size * 0.4,
            shadows: [
              if (number > 10)
                const Shadow(
                  color: Colors.black38,
                  offset: Offset(1, 1),
                  blurRadius: 3,
                )
              else
                const Shadow(
                  color: Colors.black26,
                  offset: Offset(1, 1),
                  blurRadius: 2,
                )
            ],
          ),
        ),
      ),
    );
  }
}
