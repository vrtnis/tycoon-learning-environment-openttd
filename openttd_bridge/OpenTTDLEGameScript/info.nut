class OpenTTDLEGameScriptInfo extends GSInfo {
    function GetAuthor() { return "TycoonLE OpenTTD"; }
    function GetName() { return "OpenTTDLEGameScript"; }
    function GetShortName() { return "OTLG"; }
    function GetDescription() { return "Admin-port bridge for live GPT control of OpenTTD."; }
    function GetVersion() { return 1; }
    function GetDate() { return "2026-05-14"; }
    function CreateInstance() { return "OpenTTDLEGameScript"; }
    function GetAPIVersion() { return "15"; }
}

RegisterGS(OpenTTDLEGameScriptInfo());
