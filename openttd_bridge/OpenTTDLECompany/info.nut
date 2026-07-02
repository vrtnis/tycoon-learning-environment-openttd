class OpenTTDLECompanyInfo extends AIInfo {
    function GetAuthor() { return "TycoonLE OpenTTD"; }
    function GetName() { return "OpenTTDLECompany"; }
    function GetShortName() { return "OTLC"; }
    function GetDescription() { return "Passive company holder for TycoonLE OpenTTD GameScript control."; }
    function GetVersion() { return 1; }
    function GetDate() { return "2026-05-14"; }
    function CreateInstance() { return "OpenTTDLECompany"; }
    function GetAPIVersion() { return "1.0"; }
}

RegisterAI(OpenTTDLECompanyInfo());
