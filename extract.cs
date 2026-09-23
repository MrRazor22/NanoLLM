using System.Text;

var root = args.Length > 0 ? Path.GetFullPath(args[0]) : Directory.GetCurrentDirectory();
var outDir = Path.Combine(root, "_ai_context");
Directory.CreateDirectory(outDir);

var excludedDirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase) {
    "bin", "obj", ".vs", ".git", ".vscode", ".idea", "TestResults", "node_modules", "dist", "build",
    "target", "__pycache__", ".venv", "venv", "env", ".pytest_cache", ".next", "out", "coverage", "_ai_context", "vendor", "snapshots"
};
var codeExts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) {
    ".cs", ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".c", ".cpp", ".h", ".hpp", ".java", ".kt", ".swift", ".sql"
};
var manifestNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "package.json", "Cargo.toml", "pyproject.toml", "go.mod", "setup.py" };

bool IsIgnored(string name) => name is "project_structure.md" or "solution_context.md" or "extract.cs" ||
    name.StartsWith("combined_code.") || name.EndsWith(".Designer.cs", StringComparison.OrdinalIgnoreCase) ||
    name.EndsWith(".g.cs", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".min.js", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".lock", StringComparison.OrdinalIgnoreCase);

bool IsTest(string path) => path.Contains("test", StringComparison.OrdinalIgnoreCase) || path.Contains("spec", StringComparison.OrdinalIgnoreCase);

string GetLang(string ext) => ext.ToLowerInvariant() switch {
    ".cs" => "csharp", ".py" => "python", ".rs" => "rust", ".ts" => "typescript", ".tsx" => "tsx",
    ".js" => "javascript", ".jsx" => "jsx", ".go" => "go", ".cpp" or ".c" or ".h" or ".hpp" => "cpp",
    ".java" => "java", ".kt" => "kotlin", ".swift" => "swift", ".sql" => "sql", _ => "text"
};

// Fast recursive traversal pruning excluded directories
void FindFiles(DirectoryInfo dir, List<FileInfo> manifests, List<FileInfo> code)
{
    foreach (var f in dir.EnumerateFiles()) {
        if (manifestNames.Contains(f.Name) || f.Extension.EndsWith("proj", StringComparison.OrdinalIgnoreCase)) manifests.Add(f);
        if (codeExts.Contains(f.Extension) && !IsIgnored(f.Name)) code.Add(f);
    }
    foreach (var d in dir.EnumerateDirectories())
        if (!excludedDirs.Contains(d.Name)) FindFiles(d, manifests, code);
}

var allManifests = new List<FileInfo>();
var allCodeFiles = new List<FileInfo>();
FindFiles(new DirectoryInfo(root), allManifests, allCodeFiles);

// 1. Discover projects/modules
var manifestDirs = allManifests.DistinctBy(f => f.DirectoryName).OrderBy(f => f.DirectoryName).ToList();
var targetDirs = new List<(string Name, string Path)>();
bool excludeTests = true;

if (manifestDirs.Count > 0)
{
    Console.WriteLine($"Discovered {manifestDirs.Count} Projects / Modules:\n  [A] All (exclude tests)\n  [T] All (include tests)");
    for (var i = 0; i < Math.Min(manifestDirs.Count, 30); i++) {
        var dir = manifestDirs[i].DirectoryName!;
        var name = dir == root ? Path.GetFileName(root) : Path.GetFileName(dir);
        var rel = dir == root ? "." : Path.GetRelativePath(root, dir).Replace('\\', '/');
        Console.WriteLine($"  [{i + 1}] {name,-25} ({rel} - {manifestDirs[i].Name})");
    }
    if (manifestDirs.Count > 30) Console.WriteLine($"  ... and {manifestDirs.Count - 30} more modules.");

    Console.Write("\nSelect (numbers, names, 'A', or 'T') [A]: ");
    var input = Console.ReadLine()?.Trim() ?? "A";
    excludeTests = !input.Equals("T", StringComparison.OrdinalIgnoreCase);

    if (string.IsNullOrEmpty(input) || input.Equals("A", StringComparison.OrdinalIgnoreCase))
        targetDirs.AddRange(manifestDirs.Where(m => !IsTest(m.DirectoryName!)).Select(m => (Path.GetFileName(m.DirectoryName!), m.DirectoryName!)));
    else if (input.Equals("T", StringComparison.OrdinalIgnoreCase))
        targetDirs.AddRange(manifestDirs.Select(m => (Path.GetFileName(m.DirectoryName!), m.DirectoryName!)));
    else {
        var parts = input.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var matched = manifestDirs.Where((m, idx) => parts.Any(p => (int.TryParse(p, out var n) && n == idx + 1) ||
            Path.GetFileName(m.DirectoryName!).Contains(p, StringComparison.OrdinalIgnoreCase) || m.DirectoryName!.Contains(p, StringComparison.OrdinalIgnoreCase)));
        foreach (var m in matched) targetDirs.Add((Path.GetFileName(m.DirectoryName!), m.DirectoryName!));
    }
}
if (targetDirs.Count == 0) targetDirs.Add((new DirectoryInfo(root).Name, root));

// 2. Filter code files within selected targets
var files = allCodeFiles
    .Where(f => targetDirs.Any(t => t.Path == root || f.FullName.StartsWith(t.Path, StringComparison.OrdinalIgnoreCase)) && (!excludeTests || !IsTest(f.FullName)))
    .OrderBy(f => f.FullName).ToList();

// 3. Tree generator
void AppendTree(StringBuilder sb, DirectoryInfo dir, string indent)
{
    var entries = dir.GetFileSystemInfos()
        .Where(e => e is DirectoryInfo d ? !excludedDirs.Contains(d.Name) : (codeExts.Contains(Path.GetExtension(e.Name)) && !IsIgnored(e.Name) && (!excludeTests || !IsTest(e.Name))))
        .OrderBy(e => e is FileInfo).ThenBy(e => e.Name).ToList();

    for (var i = 0; i < entries.Count; i++) {
        var isLast = i == entries.Count - 1;
        var marker = isLast ? "└── " : "├── ";
        var nextIndent = indent + (isLast ? "    " : "│   ");
        if (entries[i] is DirectoryInfo sub) { sb.AppendLine($"{indent}{marker}{sub.Name}/"); AppendTree(sb, sub, nextIndent); }
        else sb.AppendLine($"{indent}{marker}{entries[i].Name}");
    }
}

var rootDir = new DirectoryInfo(root);
var treeSb = new StringBuilder($"# Project Structure\n\n```text\n{rootDir.Name}/\n");
if (targetDirs.Count == 1 && targetDirs[0].Path == root) AppendTree(treeSb, rootDir, "");
else foreach (var t in targetDirs) { treeSb.AppendLine($"├── {t.Name}/"); AppendTree(treeSb, new DirectoryInfo(t.Path), "│   "); }
treeSb.AppendLine("```");

// 4. Build output bundles
var codeSb = new StringBuilder();
var contextSb = new StringBuilder();
var projectStats = new Dictionary<string, (int Files, int CodeLines, int TotalLines)>();

foreach (var file in files)
{
    var rel = Path.GetRelativePath(root, file.FullName).Replace('\\', '/');
    var text = await File.ReadAllTextAsync(file.FullName);
    var lines = text.Split('\n');
    var totalLines = lines.Length - (text.EndsWith('\n') ? 1 : 0);
    var codeLines = lines.Count(l => !string.IsNullOrWhiteSpace(l));

    var projName = targetDirs.FirstOrDefault(t => file.FullName.StartsWith(t.Path, StringComparison.OrdinalIgnoreCase)).Name ?? rootDir.Name;
    var cur = projectStats.GetValueOrDefault(projName);
    projectStats[projName] = (cur.Files + 1, cur.CodeLines + codeLines, cur.TotalLines + totalLines);

    var lang = GetLang(file.Extension);
    codeSb.AppendLine($"// {new string('=', 76)}\n// FILE: {rel} ({codeLines} code lines, {totalLines} total)\n// {new string('=', 76)}\n\n{text}\n");
    contextSb.AppendLine($"### File: `{rel}` ({codeLines} code lines, {totalLines} total)\n```{lang}\n{text}\n```\n");
}

var summarySb = new StringBuilder("# Solution Context\n\n> Complete codebase context and project structure for AI agent assistance.\n\n## Line Count Breakdown\n\n| Module | Files | Code Lines | Total Lines |\n|---|---|---|---|\n");
Console.WriteLine("\nLine Count Breakdown:");
var (totCode, totTotal) = (0, 0);
foreach (var (proj, stat) in projectStats) {
    totCode += stat.CodeLines; totTotal += stat.TotalLines;
    summarySb.AppendLine($"| {proj} | {stat.Files} | {stat.CodeLines:N0} | {stat.TotalLines:N0} |");
    Console.WriteLine($"  {proj,-25} : {stat.Files,3} files | {stat.CodeLines,6:N0} code lines ({stat.TotalLines:N0} total)");
}
summarySb.AppendLine($"| **Total** | **{files.Count}** | **{totCode:N0}** | **{totTotal:N0}** |\n\n");
Console.WriteLine($"  {"Total",-25} : {files.Count,3} files | {totCode,6:N0} code lines ({totTotal:N0} total)\n");

contextSb.Insert(0, summarySb.ToString() + treeSb.ToString() + "\n## Source Code\n\n");

var primaryExt = files.GroupBy(f => f.Extension).OrderByDescending(g => g.Count()).FirstOrDefault()?.Key ?? ".txt";
var combinedFileName = $"combined_code{primaryExt}";

await File.WriteAllTextAsync(Path.Combine(outDir, "project_structure.md"), treeSb.ToString());
await File.WriteAllTextAsync(Path.Combine(outDir, combinedFileName), codeSb.ToString());
await File.WriteAllTextAsync(Path.Combine(outDir, "solution_context.md"), contextSb.ToString());

Console.WriteLine($"Outputs saved in: {outDir}");
