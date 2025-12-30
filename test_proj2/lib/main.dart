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
  List<List<int>> _savedNumbers = [];

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
          _savedNumbers = jsonList.map((e) => List<int>.from(e)).toList();
        });
      } catch (e) {
        debugPrint('Error loading numbers: $e');
      }
    }
  }

  Future<void> _saveToPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final String jsonString = jsonEncode(_savedNumbers);
    await prefs.setString('lotto_numbers', jsonString);
  }

  void _addNumber(List<int> numbers) {
    setState(() {
      _savedNumbers.insert(0, numbers);
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _currentIndex == 0
          ? GeneratorScreen(onSave: _addNumber)
          : HistoryScreen(
              savedNumbers: _savedNumbers,
              onDelete: _deleteNumber,
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
  final Function(List<int>) onSave;

  const GeneratorScreen({super.key, required this.onSave});

  @override
  State<GeneratorScreen> createState() => _GeneratorScreenState();
}

class _GeneratorScreenState extends State<GeneratorScreen> {
  List<List<int>> _currentNumbers = [];
  bool _isGenerating = false;
  GenMode _selectedMode = GenMode.random;
  int _totalRounds = 1205; // 기본값, API로부터 업데이트됨

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
          
          // 서버로부터 받은 총 회차 정보 업데이트
          if (data['total_rounds'] != null) {
            setState(() {
              _totalRounds = data['total_rounds'];
            });
          }
          
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
                  title: const Text('추첨 방식 안내'),
                  content: const Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('🍀 운에 맡기기:', style: TextStyle(fontWeight: FontWeight.bold)),
                      Text('완전 무작위로 번호를 생성합니다.\n', style: TextStyle(fontSize: 13)),
                      Text('🧠 빅데이터 분석:', style: TextStyle(fontWeight: FontWeight.bold)),
                      Text('제1회부터 최신 회차까지 전체 당첨 데이터로 학습된 LSTM 딥러닝 모델(PyTorch)이 최근 20회차를 입력받아 시계열 패턴을 분석하고, 확률적으로 가중치를 적용한 번호를 추천합니다.', style: TextStyle(fontSize: 13)),
                    ],
                  ),
                  actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('닫기'))],
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
          
          const SizedBox(height: 20),
          
          if (isAnalysis)
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 0),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF1A237E),
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.indigo.withOpacity(0.3),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Row(
                children: [
                  const Icon(Icons.insights, color: Colors.amber, size: 32),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'LSTM 딥러닝 모델 활성화',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Container(
                          margin: const EdgeInsets.only(top: 4, bottom: 4),
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            '학습 데이터: 제1회 ~ 제$_totalRounds회 (PyTorch LSTM)',
                            style: const TextStyle(
                              color: Colors.amberAccent,
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        Text(
                          '전체 회차 데이터로 학습된 신경망이\n최근 20회차 패턴을 분석하여 번호를 예측합니다.',
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.8),
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  )
                ],
              ),
            ),

          Expanded(
            child: _isGenerating
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 150,
                          child: LinearProgressIndicator(
                            color: isAnalysis ? const Color(0xFF1A237E) : Colors.green,
                            backgroundColor: Colors.grey[200],
                          ),
                        ),
                        const SizedBox(height: 24),
                        Text(
                          isAnalysis 
                            ? 'LSTM 딥러닝 모델 연산 중...\n서버에서 AI 예측 번호를 생성하고 있습니다'
                            : '행운의 숫자를 뽑고 있습니다!',
                          textAlign: TextAlign.center,
                          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                        ),
                      ],
                    ),
                  )
                : _currentNumbers.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              isAnalysis ? Icons.pie_chart_outline : Icons.shuffle,
                              size: 80,
                              color: Colors.grey[300],
                            ),
                            const SizedBox(height: 16),
                            Text(
                              isAnalysis
                                  ? '빅데이터 기반으로\n당첨 확률을 높여보세요!'
                                  : '오늘의 운세를 시험해보세요!',
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.grey[500], fontSize: 16),
                            ),
                          ],
                        ),
                      )
                    : Scrollbar(
                        thumbVisibility: true,
                        thickness: 6,
                        radius: const Radius.circular(10),
                        child: ListView.builder(
                          padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
                          itemCount: _currentNumbers.length,
                          itemBuilder: (context, index) {
                            return _buildLottoRow(
                              _currentNumbers[index],
                              index + 1,
                              isAnalysis,
                            );
                          },
                        ),
                      ),
          ),
        ],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: Padding(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
        child: isAnalysis
            ? 
            // 빅데이터 분석 모드: 5게임만 생성
            SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton.icon(
                  onPressed: _isGenerating ? null : () => _generate(5),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A237E),
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.psychology, size: 28),
                  label: const Text(
                    'AI 분석 조합 5개 생성',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                ),
              )
            : 
            // 운에 맡기기 모드: 1게임 / 5게임 선택 가능
            Row(
                children: [
                  Expanded(
                    child: SizedBox(
                      height: 56,
                      child: ElevatedButton.icon(
                        onPressed: _isGenerating ? null : () => _generate(1),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: Colors.green,
                          side: const BorderSide(color: Colors.green),
                        ),
                        icon: const Icon(Icons.looks_one),
                        label: const Text('1게임', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: SizedBox(
                      height: 56,
                      child: ElevatedButton.icon(
                        onPressed: _isGenerating ? null : () => _generate(5),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
                        ),
                        icon: const Icon(Icons.filter_5),
                        label: const Text(
                          '랜덤 5게임 생성',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
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
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isAnalysis ? const Color(0xFF1A237E).withOpacity(0.1) : Colors.grey.withOpacity(0.2),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.grey[100],
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            '${index}번 게임',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                              color: Colors.grey,
                            ),
                          ),
                        ),
                        if (isAnalysis) ...[
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: Colors.amber.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: const Row(
                              children: [
                                Icon(Icons.star, size: 12, color: Colors.amber),
                                SizedBox(width: 4),
                                Text(
                                  '추천',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.deepOrange,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ]
                      ],
                    ),
                    IconButton(
                      icon: const Icon(Icons.bookmark_border_rounded, color: Colors.grey),
                      onPressed: () => widget.onSave(numbers),
                      tooltip: '저장',
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: numbers.map((n) => LottoBall(number: n)).toList(),
                ),
              ],
            ),
          ),
          if (isAnalysis)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF1A237E).withOpacity(0.05),
                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(16)),
              ),
              child: const Text(
                'LSTM 신경망 딥러닝 모델 예측 완료',
                style: TextStyle(
                  color: Color(0xFF1A237E),
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.right,
              ),
            ),
        ],
      ),
    );
  }
}

class HistoryScreen extends StatelessWidget {
  final List<List<int>> savedNumbers;
  final Function(int) onDelete;

  const HistoryScreen({
    super.key,
    required this.savedNumbers,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('저장 보관함'),
      ),
      body: savedNumbers.isEmpty
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
          : ListView.builder(
              itemCount: savedNumbers.length,
              padding: const EdgeInsets.all(16),
              itemBuilder: (context, index) {
                final numbers = savedNumbers[index];
                return Dismissible(
                  key: Key(numbers.toString() + index.toString()),
                  direction: DismissDirection.endToStart,
                  onDismissed: (_) => onDelete(index),
                  background: Container(
                    alignment: Alignment.centerRight,
                    padding: const EdgeInsets.only(right: 20),
                    color: Colors.red,
                    child: const Icon(Icons.delete, color: Colors.white),
                  ),
                  child: Card(
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                      side: BorderSide(color: Colors.grey.withOpacity(0.2)),
                    ),
                    margin: const EdgeInsets.only(bottom: 12),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                'No. ${savedNumbers.length - index}',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                  color: Colors.grey[800],
                                ),
                              ),
                              IconButton(
                                icon: const Icon(Icons.delete_outline,
                                    color: Colors.red),
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
                                            onDelete(index);
                                          },
                                          child: const Text('삭제', style: TextStyle(color: Colors.red)),
                                        ),
                                      ],
                                    ),
                                  );
                                },
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children:
                                numbers.map((n) => LottoBall(number: n)).toList(),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
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
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        color: _getBallColor(number),
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.15),
            blurRadius: 4,
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
            fontSize: 16,
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
